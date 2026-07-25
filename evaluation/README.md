# Sample-video evaluation

Run `python evaluation/evaluate_sample_videos.py` from the project root. The report verifies video readability and measures detector coverage, confidence, object-class counts, and processing time. It does not call confidence "accuracy."

To calculate precision, recall, and F1, add human-reviewed object labels to `ground_truth.json`. Presence labels measure whether each class appears anywhere in a video; frame-level and event-level accuracy require correspondingly detailed annotations and are not inferred by the simple evaluator.

Run `python evaluation/verify_e2e_workflow.py` for a real local multipart upload, metadata, stages, frames, image routes, timestamp mapping, matching query, and insufficient-evidence regression. Run `python evaluation/verify_browser_workflow.py` to drive the same workflow in installed Microsoft Edge and validate result seeking, console/network errors, and mobile overflow.
