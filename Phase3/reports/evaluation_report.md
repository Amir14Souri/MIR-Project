# Stage 4 Evaluation Report

## Metrics Summary

The table below summarizes the retrieval and reranking performance across all evaluated queries:

| Run | Query type | P@10 | R@20 | MAP | NDCG@10 | Notes |
|---|---|---:|---:|---:|---:|---|
| Dense single-vector multimodal | all | [TODO] | [TODO] | [TODO] | [TODO] | image+text product vectors |
| Sparse SPLADE++ | text/image_text | [TODO] | [TODO] | [TODO] | [TODO] | text-bearing queries only |
| Hybrid prefusion | text/image_text | [TODO] | [TODO] | [TODO] | [TODO] | dense + sparse |
| Hybrid + cross-encoder | all | [TODO] | [TODO] | [TODO] | [TODO] | final required run |
| Hybrid + cross-encoder + retrieval-score fusion | all | [TODO] | [TODO] | [TODO] | [TODO] | final fused run |

## Analysis
[TODO: Write a brief summary comparing the different approaches. Discuss whether hybrid retrieval improved candidate pools over dense/sparse alone, and whether the cross-encoder improved final rankings.]

## Judged Coverage
[TODO: Report the percentage of returned products that have qrels.]
