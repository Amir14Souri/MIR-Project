# Stage 2 + Stage 3 run

Run timestamp: 20260725T213421Z
GPU:          Tesla T4
Queries:      20

## What ran
- Stage 2 (hybrid retrieval: dense Qwen3-VL + sparse SPLADE, RRF prefusion)
- Stage 3 (cross-encoder rerank with BGE-reranker-v2-m3, z-score fusion)

## Top-1 by query type (reranked output)
- text: 10 queries, top-1 -> B086B5MRFB|amazon.in
- image: 5 queries, top-1 -> B086TGRSBG|amazon.com
- image_text: 5 queries, top-1 -> B07Y5XQ2LS|amazon.in

## Files
- outputs/retrieval_results.jsonl
- outputs/reranked_results.jsonl
- outputs/reranked_strategy_a.jsonl
- reports/retrieval_summary.md
- reports/reranking_summary.md