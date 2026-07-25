# Evidence and knowledge model

`VideoAsset` is immutable metadata keyed by SHA-256. `Evidence` is the atomic fact record with video, time interval, confidence, producer, and optional frame, track, object, and event identifiers. `CandidateEvidence` adds retrieval scores without mutating facts. `VerifiedEvidence` records an accept or reject decision. `GroundedAnswer` refuses a verified state when citations are absent.

Graph nodes represent entities, locations, and events. Edges represent entered, left, carried, approached, crossed, before, after, and during. Every edge requires evidence IDs, so graph relations are indexes over evidence rather than independent claims.
