# Stage 1 - Indexing Summary

## Dense embedding
- Model: `Qwen/Qwen3-VL-Embedding-2B`
- Loading config: device=cuda, torch_dtype=torch.float16
- Encode method: encode_document (input = {'text': product_text, 'image': image_path})
- Dense dimension: 2048
- Batch size: 2
- Runtime: Tesla T4
- Vectors produced: 2000 (normalized: True)

## Sparse embedding
- Model: `prithivida/Splade_PP_en_v2`
- Max sequence length: 256
- Vocab size: 30522
- Top-N kept per product: 128
- Avg / min / max non-zero count: 120.4 / 46 / 128

## Indexes
- Dense: FAISS `IndexFlatIP` — cosine via inner product on normalized vectors.
- Sparse: SciPy CSR matrix (`sparse_index.npz`) — dot-product search.
- Vector store: faiss_plus_csr
- ID mapping file: `artifacts/embeddings/product_ids.txt` (row i <-> product_id).

## Sanity-check searches
- DENSE image-query (product 'B07B51H7B1|amazon.com'): self rank=1, top=['B086TGRSBG|amazon.com', 'B07B51H7B1|amazon.com', 'B07B4W2MFW|amazon.com']
- DENSE text-query (product 'B07SYL8MPN|amazon.sa'): self rank=0, top=['B07SYL8MPN|amazon.sa', 'B07T4BMN8Z|amazon.co.uk', 'B07T25P4BB|amazon.com']
- DENSE image+text (product 'B075X25BYC|amazon.com'): self rank=0, top=['B075X25BYC|amazon.com', 'B07QX2BYDH|amazon.com', 'B07QW3JRT4|amazon.com']
- SPARSE title-query (product 'B07B51H7B1|amazon.com'): self rank=0, top=['B07B51H7B1|amazon.com', 'B075X2CVQK|amazon.com', 'B07CVC1Q7C|amazon.com']
- SPARSE title-query (product 'B07SYL8MPN|amazon.sa'): self rank=0, top=['B07SYL8MPN|amazon.sa', 'B07SYL958P|amazon.sg', 'B07T25P4BB|amazon.com']
- SPARSE title-query (product 'B075X25BYC|amazon.com'): self rank=0, top=['B075X25BYC|amazon.com', 'B07QX2BYDH|amazon.com', 'B08DWMY83P|amazon.com']
