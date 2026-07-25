# Sample-video evaluation

Run `python evaluation/evaluate_sample_videos.py` from the project root. The report verifies video readability and measures detector coverage, confidence, object-class counts, and processing time. It does not call confidence “accuracy.”

To calculate precision, recall, and F1, add human-reviewed object labels to `ground_truth.json`, for example `{"expected_objects_by_video":{"asset3.mp4":["person","umbrella"]}}`. Presence labels measure whether each class appears anywhere in a video; frame-level and event-level accuracy require correspondingly detailed annotations and are not inferred by this simple evaluator.
