# Evaluation

Evaluation is review-gated. `ground_truth.json` intentionally contains no labels until a human reviewer adds them; an empty schema is more truthful than fabricated accuracy.

Each reviewed video entry must use `review_status: "reviewed"`, `present_objects`, timestamped `segments`, and query cases with accepted timestamp windows. The evaluator reports video-level presence precision, recall, and F1 only for reviewed entries. Temporal IoU, retrieval recall@K, verification precision, and abstention quality require the corresponding query/segment annotations and must be reported separately.

Run `python evaluation/evaluate_sample_videos.py` for readability, coverage, confidence distributions, and reviewed presence metrics. Its output is generated under `output/evaluation/`, never committed. Run `python evaluation/verify_e2e_workflow.py` for the local multipart upload-to-query contract. The browser workflow additionally requires its documented optional dependency.
