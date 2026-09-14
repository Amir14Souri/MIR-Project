# Stage 3: Reranking Summary

            ## Pipeline Overview
            - **Reranker Model**: BAAI/bge-reranker-v2-m3
            - **Pair Format**: Query text (or surrogate) + Product metadata (Title, Category, Brand, Color, Material, etc. joined by ' | ')
            - **Reranking Depth**: Top 100 candidates from the Hybrid search stage
            - **Score Normalization**: Z-score normalization for Dense, Sparse, and CE scores prior to final fusion.
            - **Fusion Weights**:
            - Text queries: {'ce': 0.4, 'dense': 0.3, 'sparse': 0.3}
            - Image queries: {'ce': 0.0, 'dense': 1.0, 'sparse': 0.0}
            - Multimodal queries: {'ce': 0.3, 'dense': 0.4, 'sparse': 0.3}

            ## Query Breakdown
            - **Text queries**: 10
            - **Image queries**: 5
            - **Image+Text queries**: 5

            ## Examples: Before & After Reranking
            ### TEXT Example (Standard)
- **Query**: gray fabric sofa for a small living room, not leather
- **Hybrid Top-1 (Before)**: B086B5MRFB|amazon.in
- **Reranked Top-1 (After)**: B086B5MRFB|amazon.in

### IMAGE Example (Standard)
- **Query**: data/raw/images/small/ce/ce04dd2a.jpg
- **Hybrid Top-1 (Before)**: B086TGRSBG|amazon.com
- **Reranked Top-1 (After)**: B086TGRSBG|amazon.com

### IMAGE_TEXT Example (Negative Constraint)
- **Query**: similar style, but in black and not for dining
- **Hybrid Top-1 (Before)**: B07Y5XQ2LS|amazon.in
- **Reranked Top-1 (After)**: B07Y5XQ2LS|amazon.in

