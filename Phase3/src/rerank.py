import argparse
import os
import json
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default="data/catalog_subset.parquet")
    parser.add_argument("--queries", type=str, default="data/queries.jsonl")
    parser.add_argument("--retrieval", type=str, default="outputs/retrieval_results.jsonl")
    parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank_top_n", type=int, default=100)
    parser.add_argument("--out", type=str, default="outputs/reranked_results.jsonl")
    return parser.parse_args()

def zscore(vals):
    if len(vals) == 0: return []
    arr = np.array(vals, dtype=np.float32)
    mu, sd = float(arr.mean()), float(arr.std() + 1e-9)
    return [float((v - mu) / sd) for v in arr]

def _safe(v, default=''):
    if v is None: return default
    if isinstance(v, float) and math.isnan(v): return default
    return v

def build_product_text(row, max_chars=1024):
    fields = {
        "Title": _safe(getattr(row, 'title', None)),
        "Category": _safe(getattr(row, 'category_path', None)),
        "Product type": _safe(getattr(row, 'product_type', None)),
        "Brand": _safe(getattr(row, 'brand', None)),
        "Color": _safe(getattr(row, 'color', None)),
        "Material": _safe(getattr(row, 'material', None)),
        "Style": _safe(getattr(row, 'style', None)),
        "Features": _safe(getattr(row, 'bullet_points', None)),
        "Description": _safe(getattr(row, 'description', None)),
    }
    parts = [f"{k}: {v}" for k, v in fields.items() if v]
    text = " | ".join(parts)
    return text[:max_chars]

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    print(f"[rerank] loading catalog {args.catalog}")
    catalog = pd.read_parquet(args.catalog)
    cat_by_id = {r.product_id: r for r in catalog.itertuples()}
    
    print(f"[rerank] loading queries {args.queries}")
    queries_by_id = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries_by_id[q["query_id"]] = q
                
    print(f"[rerank] loading retrieval results {args.retrieval}")
    out_records = []
    with open(args.retrieval, encoding="utf-8") as f:
        for line in f:
            if line.strip(): out_records.append(json.loads(line))
            
    print(f"[rerank] loading cross-encoder {args.reranker}")
    import torch
    from sentence_transformers import CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ce_model = CrossEncoder(args.reranker, max_length=512, default_activation_function=torch.nn.Identity())
    ce_model.model.to(device)
    
    WEIGHTS = {
        "text": {"ce": 0.4, "dense": 0.3, "sparse": 0.3},
        "image": {"ce": 0.0, "dense": 1.0, "sparse": 0.0},
        "image_text": {"ce": 0.3, "dense": 0.4, "sparse": 0.3},
    }
    
    pairs_per_query = []
    for r in tqdm(out_records, desc="build pairs"):
        qid = r["query_id"]
        q = queries_by_id[qid]
        qtext = 'find similar item' if q["query_type"] == "image" else q["query_text"]
        hits = r["results"][:args.rerank_top_n]
        pairs = []
        for h in hits:
            row = cat_by_id.get(h["product_id"])
            if row is not None:
                pairs.append((qtext, build_product_text(row)))
        pairs_per_query.append((qid, q["query_type"], pairs, hits))
        
    rerank_records = []
    t0 = time.time()
    for qid, qtype, pairs, hits in tqdm(pairs_per_query, desc="rerank"):
        weights = WEIGHTS[qtype]
        ce_raw = ce_model.predict(pairs, batch_size=32) if pairs else []
        d_raw = [h.get("dense_score", 0.0) for h in hits]
        s_raw = [h.get("sparse_score", 0.0) if h.get("sparse_score") is not None else 0.0 for h in hits]
        
        d_z = zscore(d_raw)
        s_z = zscore(s_raw)
        c_z = zscore(ce_raw)
        
        scored_hits = []
        for i, h in enumerate(hits):
            final_score = (d_z[i] * weights["dense"] + s_z[i] * weights["sparse"] + c_z[i] * weights["ce"])
            scored_hits.append({
                "product_id": h["product_id"],
                "final_score": final_score,
                "cross_encoder_score": float(ce_raw[i]) if len(ce_raw) > 0 else 0.0,
                "dense_score": h.get("dense_score"),
                "sparse_score": h.get("sparse_score"),
                "prefusion_score": h.get("prefusion_score"),
                "source": h.get("source")
            })
            
        scored_hits.sort(key=lambda x: x["final_score"], reverse=True)
        for rank, h in enumerate(scored_hits, 1): h["rank"] = rank
        
        rerank_records.append({
            "query_id": qid,
            "query_type": qtype,
            "run_name": "hybrid_cross_encoder_fusion",
            "reranker_type": "text_cross_encoder",
            "results": scored_hits
        })
        
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rerank_records: f.write(json.dumps(r) + "\n")
    print(f"[rerank] wrote {len(rerank_records)} records to {args.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
