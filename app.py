import os

import gradio as gr

from pipeline import VisionGuardPipeline


pipe = VisionGuardPipeline()


def _cls(mode):
    mp = {
        "all": None,
        "person": [0],
        "vehicle": [1, 2, 3, 5, 7],
    }
    return mp.get(mode, None)


def _fmt_meta(meta):
    return (
        f"video: `{os.path.basename(meta['video'])}`\n\n"
        f"- duration: `{meta['duration']:.2f}s`\n"
        f"- fps: `{meta['fps']:.2f}`\n"
        f"- sampled every: `{meta['sample_sec']:.2f}s`\n"
        f"- search windows: `{meta['segments']}`"
    )


def build_idx(video, mode, sample_sec, win_sec, progress=gr.Progress()):
    if not video:
        return "upload a video first", None, None
    res = pipe.index_video(video, cls=_cls(mode), sample_sec=sample_sec, win_sec=win_sec, progress=progress)
    return _fmt_meta(res["meta"]), res["index_json"], gr.update(interactive=True)


def run_search(q, top_k, clip_pad):
    if not pipe.idx:
        return "index a video first", [], None, None, None, None
    if not q or not q.strip():
        return "enter a natural-language query", [], None, None, None, None
    hits = pipe.search(q.strip(), top_k=int(top_k))
    exp = pipe.export_hits(q.strip(), hits, clip_pad=clip_pad)
    rows = []
    md = [f"## matches for `{q}`", ""]
    if not exp["hits"]:
        md.append("no strong matches found")
    for i, x in enumerate(exp["hits"], 1):
        rows.append([i, round(x["score"], 4), round(x["start"], 2), round(x["end"], 2), x["summary"], ", ".join(x["objects"])])
        md.append(f"{i}. `{x['start']:.2f}s - {x['end']:.2f}s` score `{x['score']:.4f}`  ")
        md.append(f"   {x['summary']}")
    first = exp["hits"][0]["clip"] if exp["hits"] else None
    clips = [x["clip"] for x in exp["hits"]]
    files = [exp["html"], exp["csv"], exp["json"], exp["zip"]]
    return "\n".join(md), rows, first, clips, files, exp["json"]


with gr.Blocks(title="VisionGuard AI") as demo:
    gr.Markdown(
        """
# VisionGuard AI

Index CCTV footage, search it with natural language, and export matched clips with timestamp records.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="cctv video")
            mode = gr.Radio(["all", "person", "vehicle"], value="all", label="focus")
            sample_sec = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="sample every (sec)")
            win_sec = gr.Slider(2.0, 15.0, value=6.0, step=1.0, label="match window (sec)")
            idx_btn = gr.Button("index video", variant="primary")
            idx_md = gr.Markdown()
            idx_file = gr.File(label="index json")

            gr.Markdown("### query")
            q = gr.Textbox(placeholder="person sitting near gate, white car entering, group at entrance")
            top_k = gr.Slider(1, 10, value=5, step=1, label="top clips")
            clip_pad = gr.Slider(0.0, 6.0, value=2.0, step=0.5, label="clip padding (sec)")
            src = ["assets/asset1.mp4", "assets/asset2.mp4", "assets/asset3.mp4"]
            good = [x for x in src if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            search_btn = gr.Button("search and export", interactive=False)

        with gr.Column(scale=2):
            out_md = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["rank", "score", "start", "end", "summary", "objects"], interactive=False)
            out_vid = gr.Video(label="top clip")
            out_clips = gr.Files(label="all clips")
            out_files = gr.Files(label="reports")
            raw_json = gr.File(label="raw search json")

    idx_btn.click(build_idx, [video, mode, sample_sec, win_sec], [idx_md, idx_file, search_btn])
    search_btn.click(run_search, [q, top_k, clip_pad], [out_md, out_tbl, out_vid, out_clips, out_files, raw_json])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG"))
    demo.launch(server_name="0.0.0.0", share=share, show_error=True)
