# Bundled sample videos

These videos are inputs for functional checks. Their original filenames are retained because tests and UI sample selection refer to them. Descriptions below are detector-derived summaries, not human ground-truth annotations.

| File | Duration | Suggested human description | Dominant sampled detections |
|---|---:|---|---|
| `asset1.mp4` | 299.0s | Indoor people and seating scene | person, chair, TV, suitcase, handbag |
| `asset2.mp4` | 17.0s | People near trucks and cars | person, truck, car, chair |
| `asset3.mp4` | 11.9s | Indoor person and screen scene | person, TV; one low-confidence umbrella detection |
| `asset4.mp4` | 47.7s | Road traffic with cars and buses | car, bus, person, truck |
| `asset5.mp4` | 29.6s | Busy mixed road traffic | car, truck, person, motorcycle, bus |
| `asset6.mp4` | 77.3s | Crowded street and traffic | person, car, umbrella, suitcase, truck |
| `asset7.mp4` | 30.0s | Cars at traffic lights | car, traffic light, person, truck |
| `asset8.mp4` | 27.0s | Cars and motorcycles at traffic lights | car, traffic light, person, truck, motorcycle |

Run `python evaluation/evaluate_sample_videos.py` to regenerate the detector report. Add reviewed labels to `evaluation/ground_truth.json` before interpreting precision, recall, or F1 as accuracy metrics.
