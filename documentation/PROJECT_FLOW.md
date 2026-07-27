# VisionGuard Project Flow

> **Maintenance rule:** This is the canonical operational-flow document. Any change to video ingestion, indexing, deduplication, retrieval, verification, or evidence export must update this file in the same pull request.

## Exact flow

1. The user uploads an MP4 or selects a bundled video. Upload stores the original file and decoder metadata only; it does not start indexing.
2. The user explicitly starts indexing. The pipeline opens the video with Decord and reads every source frame in ascending frame-number order.
3. Each decoded frame is compared with the immediately preceding decoded frame. It is removed only if its shape, data type, and every pixel are identical. There is no time-based sampling, motion threshold, forced keyframe interval, empty-frame removal, or decoder fallback.
4. Every retained frame is submitted for source-evidence writing and processed once by YOLO with BoT-SORT tracking. That single inference returns every detector box and adds a track ID where available; no second YOLO detection pass is run.
5. Evidence-frame writes run concurrently with later CPU work. Retained frames are grouped by source timestamps, so deduplication never stretches a `WIN_SEC` segment.
6. After an authenticated readiness probe, NVIDIA analyzes each representative segment frame. Structured output is stored as an unverified `semantic_description` tied to detector context and a source frame. Failure is terminal; there is no semantic fallback.
7. The event extractor builds first-observed state, measured movement, observed dwell, and configured zone-transition `event_fact` records. It does not call last observation a disappearance or stationary state movement.
8. LangGraph routes object, distinct-track count, numeric temporal, implemented event, zone, vector semantic-scene, and explicit-verification requests. Unsupported events abstain rather than falling through to generic semantics.
9. Explicit verification requires an authenticated reachable verifier. Rejected, unavailable, malformed, or missing verification abstains. Ordinary search never invokes verification implicitly.
10. The UI/API returns either evidence frames, a timestamped textual answer, or both. Exported clips and reports remain traceable to the stored source evidence.

```mermaid
flowchart TD
    A[Upload MP4 or choose bundled video] --> B[Store original video and decoder metadata]
    B --> C[User explicitly starts indexing]
    C --> D[Decord decodes frame 0 through frame N in order]
    D --> E{Exactly equal to immediately previous decoded frame?}
    E -- Yes --> F[Discard exact consecutive duplicate]
    E -- No --> G[Write retained source evidence frame]
    G --> H[One YOLO plus BoT-SORT inference per retained frame]
    H --> I[Keep every detector box; attach track ID when available]
    I --> J[Store frame number timestamp boxes confidence classes tracks appearances]
    J --> K[SigLIP frame embedding]
    K --> L[Build frame vector index]
    J --> M[Group retained frames into WIN_SEC evidence segments]
    M --> N[Build segment vector index]
    L --> O[Mark video searchable]
    N --> O

    P[User natural-language query] --> Q[Deterministic query normalization]
    Q --> R[Resolve runtime labels and documented aliases]
    R --> S{Evidence route}
    S --> T[Detector and track evidence retrieval]
    S --> U[Semantic frame or segment retrieval]
    S --> V[Unsupported request: abstain with limitation]
    T --> W[Calibrate and group into evidence segments]
    U --> X[Rank stored evidence segments]
    W --> Y[Optional explicitly configured visual verification]
    X --> Y
    Y --> Z[Frames text answer or both with exact timestamps]
    V --> Z
    Z --> AA[Optional evidence clip and report export]
```

## Non-negotiable indexing invariants

- Every source frame is decoded in order.
- Only exact consecutive pixel duplicates are discarded.
- Every retained frame has a source frame number and decoder-derived timestamp.
- Every retained frame has one YOLO inference; untracked detector boxes are retained instead of triggering a second detection pass.
- No retrieved result is presented as visually verified unless an explicitly configured verifier confirms it.
- `WIN_SEC` affects segment aggregation only; it never changes which frames are extracted or indexed.
