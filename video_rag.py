import os
import subprocess
import sys
from pathlib import Path


root = Path.cwd()
fd = root / "frames"
cd = root / "clips"
dbd = root / "db"
clip = None
wh = None
cli = None
fc = None
tc = None


def run(cmd):
    subprocess.run(cmd, check=True)


def ff():
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def model():
    global clip
    if clip is None:
        from sentence_transformers import SentenceTransformer

        clip = SentenceTransformer("clip-ViT-L-14", device=device())
    return clip


def whisper():
    global wh
    if wh is None:
        from faster_whisper import WhisperModel

        dev = device()
        dt = "float16" if dev == "cuda" else "int8"
        wh = WhisperModel("large-v3", device=dev, compute_type=dt)
    return wh


def cols():
    global cli, fc, tc
    if cli is None:
        import chromadb

        dbd.mkdir(exist_ok=True)
        cli = chromadb.PersistentClient(path=str(dbd))
        fc = cli.get_or_create_collection("frames", metadata={"hnsw:space": "cosine"})
        tc = cli.get_or_create_collection("transcripts", metadata={"hnsw:space": "cosine"})
    return fc, tc


def enc_img(p):
    from PIL import Image

    emb = model().encode(Image.open(p).convert("RGB"), normalize_embeddings=True)
    return emb.tolist()


def enc_txt(x):
    emb = model().encode(str(x), normalize_embeddings=True)
    return emb.tolist()


def vid_id(p):
    return Path(p).stem


def scene_rows(vp):
    from scenedetect import ContentDetector, detect

    sc = detect(str(vp), ContentDetector())
    if sc:
        return [(a.get_seconds(), b.get_seconds()) for a, b in sc if b.get_seconds() > a.get_seconds()]
    out = subprocess.check_output([
        os.environ.get("FFPROBE_BIN", "ffprobe"), "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(vp),
    ], text=True).strip()
    return [(0.0, float(out))]


def frame_at(vp, t, p):
    run([
        ff(), "-y", "-ss", f"{t:.3f}", "-i", str(vp),
        "-frames:v", "1", "-q:v", "2", str(p),
    ])


def trim(vp, st, en, p):
    run([
        ff(), "-y", "-ss", f"{st:.3f}", "-to", f"{en:.3f}", "-i", str(vp),
        "-c", "copy", "-avoid_negative_ts", "make_zero", str(p),
    ])


def ingest(vid):
    fd.mkdir(exist_ok=True)
    cd.mkdir(exist_ok=True)
    dbd.mkdir(exist_ok=True)
    vp = Path(vid).resolve()
    vn = vid_id(vp)
    rows = scene_rows(vp)
    fcol, tcol = cols()
    ids, embs, metas, docs = [], [], [], []
    for i, (st, en) in enumerate(rows):
        fp = fd / f"{vn}_{i:05d}.jpg"
        frame_at(vp, (st + en) / 2, fp)
        ids.append(f"{vn}:frame:{i:05d}")
        embs.append(enc_img(fp))
        metas.append({"vid": vn, "vpath": str(vp), "type": "frame", "st": st, "en": en, "fpath": str(fp)})
        docs.append(f"visual scene {vn} {st:.2f}-{en:.2f}")
    if ids:
        fcol.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
    segs, _ = whisper().transcribe(str(vp), vad_filter=True)
    ids, embs, metas, docs = [], [], [], []
    for i, s in enumerate(segs):
        txt = s.text.strip()
        if not txt:
            continue
        ids.append(f"{vn}:txt:{i:05d}")
        embs.append(enc_txt(txt))
        metas.append({"vid": vn, "vpath": str(vp), "type": "transcript", "st": float(s.start), "en": float(s.end), "fpath": "", "text": txt})
        docs.append(txt)
    if ids:
        tcol.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=docs)
    print(f"ingested {vn}: {len(rows)} frames, {len(ids)} transcripts")


def qcol(col, qe, k, vf):
    whr = {"vid": vf} if vf else None
    res = col.query(query_embeddings=[qe], n_results=k, where=whr)
    out = []
    for i, mid in enumerate(res["ids"][0]):
        mt = res["metadatas"][0][i]
        mt["id"] = mid
        mt["score"] = float(res["distances"][0][i])
        mt["source"] = mt["type"]
        mt["doc"] = res["documents"][0][i] if res.get("documents") else ""
        out.append(mt)
    return out


def search(q, k=5, vid_filter=None):
    fcol, tcol = cols()
    qe = enc_txt(q)
    it = qcol(fcol, qe, k, vid_filter) + qcol(tcol, qe, k, vid_filter)
    it.sort(key=lambda x: x["score"])
    return it[:k]


def get_assets(items):
    cd.mkdir(exist_ok=True)
    out = []
    for i, it in enumerate(items):
        vp = it["vpath"]
        st, en = float(it["st"]), float(it["en"])
        p = cd / f"{Path(vp).stem}_{i:02d}_{st:.2f}_{en:.2f}.mp4"
        trim(vp, st, en, p)
        print(f"{p} | {it['source']} | score={it['score']:.4f} | {st:.2f}-{en:.2f}")
        if it["source"] == "frame":
            print(f"frame: {it.get('fpath', '')}")
        else:
            print(f"text: {it.get('text', '')[:180]}")
        out.append(str(p))
    return out


def rag_answer(q, items):
    import requests

    ctx = []
    for it in items:
        st, en = float(it["st"]), float(it["en"])
        if it["source"] == "transcript":
            ctx.append(f"{st:.2f}-{en:.2f} transcript: {it.get('text', '')}")
        else:
            ctx.append(f"{st:.2f}-{en:.2f} visual scene: {it.get('fpath', '')}")
    prm = "Answer only from this video evidence.\nQuestion: " + q + "\nEvidence:\n" + "\n".join(ctx)
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1", "prompt": prm, "stream": False, "options": {"temperature": 0.3}},
        timeout=120,
    )
    return res.json().get("response", "").strip()


def main():
    if len(sys.argv) < 3:
        print("usage: python video_rag.py ingest video.mp4 | python video_rag.py query words")
        return
    cmd = sys.argv[1].strip().lower()
    if cmd == "ingest":
        ingest(sys.argv[2])
        return
    if cmd == "query":
        q = " ".join(sys.argv[2:])
        items = search(q)
        for i, it in enumerate(items, 1):
            print(f"{i}. {it['source']} score={it['score']:.4f} {it['vid']} {float(it['st']):.2f}-{float(it['en']):.2f}")
        get_assets(items)
        print("RAG Answer:")
        print(rag_answer(q, items))
        return
    print("unknown command")


if __name__ == "__main__":
    main()
