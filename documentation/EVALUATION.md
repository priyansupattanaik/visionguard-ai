# Evaluation methodology

A benchmark item contains a video digest, question, expected answerability, accepted timestamp ranges, known entity IDs, and prohibited claims. Report retrieval recall at K, verification precision and recall, citation temporal intersection-over-union, identity and count accuracy, abstention precision, grounded accuracy, indexing throughput, and query/cache latency.

Hallucination tests cover absent objects, ambiguous attributes, low-confidence detections, conflicting tracks, temporal inversions, unresolved co-reference, audio claims without audio evidence, and questions that presuppose nonexistent events. Correct abstention is a success.
