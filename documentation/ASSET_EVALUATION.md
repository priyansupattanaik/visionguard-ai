# Bundled asset evaluation

Evaluation was run on all eight MP4 files with YOLO11n on CPU, sampling one frame every two seconds. All eight files decoded successfully. The evaluator processed 272 frames, found one or more detections in 264 frames, and therefore measured 97.06% detection coverage. It produced 2,293 detections with a detection-count-weighted mean confidence of 0.5004. Coverage means that the detector returned something on a sampled frame; it is not accuracy.

Per-video coverage was 100% for assets 1-5, 7, and 8, and 79.49% for asset 6. Per-video mean detection confidence ranged from 0.4012 to 0.6656. The full machine-readable measurements are in `evaluation/latest_asset_report.json`.

Accuracy is intentionally `null` because these videos do not contain human ground-truth annotations. Confidence measures model certainty, not correctness. Populate `evaluation/ground_truth.json` with reviewed labels and rerun the evaluator to calculate object-presence precision, recall, and F1. Event-level and temporal-question accuracy require event and timestamp annotations rather than object-presence labels.

The latest operational regression used `asset7.mp4` on CPU with a two-second sample interval. It completed in 8.2 seconds, kept 15 frames, produced 15 nonzero 256-dimensional metadata vectors, and discovered `car`, `person`, `truck`, and `airplane` from the detector at runtime. The plural query `find the cars` resolved to the runtime label `car` and returned four detector-grounded time ranges. This is an end-to-end functionality check, not an accuracy measurement.
