import argparse
import os
import json
import time
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import faiss
from tqdm import tqdm
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default="data/catalog_subset.parquet")
    parser.add_argument("--queries", type=str, default="data/queries.jsonl")
    parser.add_argument("--index_dir", type=str, default="artifacts/index")
    parser.add_argument("--emb_dir", type=str, default="artifacts/embeddings")
    parser.add_argument("--top_k_dense", type=int, default=100)
    parser.add_argument("--top_k_sparse", type=int, default=100)
    parser.add_argument("--out", type=str, default="outputs/retrieval_results.jsonl")
    parser.add_argument("--dense_model", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--sparse_model", type=str, default="prithivida/Splade_PP_en_v2")
    parser.add_argument("--prefusion", type=str, default="rrf", choices=["rrf", "weighted"])
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--w_dense", type=float, default=0.6)
    parser.add_argument("--w_sparse", type=float, default=0.4)
    parser.add_argument("--cache_dir", type=str, default="artifacts/query_cache")
    return parser.parse_args()

def normalize_scores(pairs):
    if not pairs: return {}
    vals = np.array([s for _, s in pairs], dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        return {i: 0.0 for i, _ in pairs}
    return {i: float((s - lo) / (hi - lo)) for i, s in pairs}

def dense_search(di, q_vec, top_k):
    q = q_vec.reshape(1, -1).astype(np.float32)
    scores, idx = di.search(q, top_k)
    return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

def sparse_search(si, q_idx, q_val, top_k):
    if q_idx is None or len(q_idx) == 0: return []
    v = np.zeros(si.shape[1], dtype=np.float32)
    v[np.asarray(q_idx, dtype=np.int64)] = np.asarray(q_val, dtype=np.float32)
    scores = np.asarray(si.dot(v)).ravel()
    if top_k >= len(scores):
        top = np.argsort(-scores)
    else:
        top = np.argpartition(-scores, top_k)[:top_k]
        top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top]

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print("[retrieve] loading catalog")
    catalog = pd.read_parquet(args.catalog)
    
    print(f"[retrieve] loading dense index from {args.index_dir}/dense_index.faiss")
    dense_index = faiss.read_index(os.path.join(args.index_dir, "dense_index.faiss"))
    
    print(f"[retrieve] loading sparse index from {args.index_dir}/sparse_index.npz")
    sparse_index = sp.load_npz(os.path.join(args.index_dir, "sparse_index.npz"))
    
    with open(os.path.join(args.emb_dir, "product_ids.txt"), encoding="utf-8") as f:
        product_ids = [l.strip() for l in f if l.strip()]
        
    print(f"[retrieve] loading queries from {args.queries}")
    queries = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            q = json.loads(line)
            if q.get("query_image_path"):
                q["abs_image_path"] = q["query_image_path"] if os.path.isabs(q["query_image_path"]) else os.path.abspath(q["query_image_path"])
                if not os.path.exists(q["abs_image_path"]):
                    print(f"Warning: query {q['query_id']} image {q['abs_image_path']} not found. Skipping.")
                    continue
            queries.append(q)
            
    dense_cache_file = os.path.join(args.cache_dir, "dense_queries.npz")
    dense_cached = {}
    if os.path.isfile(dense_cache_file):
        with np.load(dense_cache_file, allow_pickle=False) as z:
            for k in z.files: dense_cached[k] = z[k]
            
    sparse_cache_file = os.path.join(args.cache_dir, "sparse_queries.jsonl")
    sparse_cached = {}
    if os.path.isfile(sparse_cache_file):
        with open(sparse_cache_file, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("indices"):
                    sparse_cached[rec["query_id"]] = (np.asarray(rec["indices"], dtype=np.int32), np.asarray(rec["values"], dtype=np.float32))
                    
    missing_dense = [q for q in queries if q["query_id"] not in dense_cached]
    if missing_dense:
        print(f"[retrieve] encoding {len(missing_dense)} dense query vectors (requires GPU/model)")
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        dense_model = SentenceTransformer(args.dense_model, model_kwargs={"torch_dtype": dtype})
        dense_model.to(device)
        fn = getattr(dense_model, "encode_query", None)
        
        for q in tqdm(missing_dense, desc="dense encode"):
            if q["query_type"] == "text": inp = q["query_text"]
            elif q["query_type"] == "image": inp = load_image(q["abs_image_path"])
            else: inp = {"text": q["query_text"], "image": load_image(q["abs_image_path"])}
            
            kw = dict(convert_to_numpy=True, normalize_embeddings=True)
            res = fn([inp], **kw) if fn is not None else dense_model.encode([inp], **kw)
            vec = res[0].astype(np.float32) if len(res.shape) > 1 else res.astype(np.float32)
            dense_cached[q["query_id"]] = vec
            
        np.savez(dense_cache_file, **dense_cached)
        
    missing_sparse = [q for q in queries if q["query_type"] in ("text", "image_text") and q["query_id"] not in sparse_cached]
    if missing_sparse:
        print(f"[retrieve] encoding {len(missing_sparse)} sparse query vectors (requires GPU/model)")
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sp_tokenizer = AutoTokenizer.from_pretrained(args.sparse_model)
        sp_model = AutoModelForMaskedLM.from_pretrained(args.sparse_model).to(device).eval()
        
        BATCH = 16
        new_records = []
        for s in tqdm(range(0, len(missing_sparse), BATCH), desc="sparse encode"):
            batch_q = missing_sparse[s:s+BATCH]
            texts = [q["query_text"] for q in batch_q]
            enc = sp_tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = sp_model(**enc)
                weights = torch.log(1 + torch.relu(outputs.logits))
                weights = weights * enc['attention_mask'].unsqueeze(-1)
                term_weights = torch.max(weights, dim=1)[0]
                for j in range(term_weights.shape[0]):
                    row_weights = term_weights[j]
                    k = min(128, (row_weights > 0).sum().item())
                    if k == 0:
                        idx, val = np.array([], dtype=np.int64), np.array([], dtype=np.float32)
                    else:
                        topk = torch.topk(row_weights, k)
                        idx, val = topk.indices.cpu().numpy(), topk.values.cpu().numpy()
                    
                    q_id = batch_q[j]["query_id"]
                    sparse_cached[q_id] = (idx, val)
                    new_records.append({'query_id': q_id, 'indices': idx.tolist(), 'values': val.tolist()})
                    
        with open(sparse_cache_file, "a", encoding="utf-8") as f:
            for rec in new_records: f.write(json.dumps(rec) + "\n")
            
    print("[retrieve] running hybrid retrieval")
    out_records = []
    t0 = time.time()
    for q in tqdm(queries, desc="retrieval"):
        qid = q["query_id"]
        qtype = q["query_type"]
        sparse_applicable = qtype in ("text", "image_text")
        
        d_hits = dense_search(dense_index, dense_cached[qid], args.top_k_dense) if qid in dense_cached else []
        s_hits = []
        if sparse_applicable and qid in sparse_cached:
            qidx, qval = sparse_cached[qid]
            s_hits = sparse_search(sparse_index, qidx, qval, args.top_k_sparse)
            
        d_norm = normalize_scores(d_hits)
        s_norm = normalize_scores(s_hits) if sparse_applicable else {}
        d_rank = {i: r for r, (i, _) in enumerate(d_hits)}
        s_rank = {i: r for r, (i, _) in enumerate(s_hits)}
        d_score_map = {i: s for i, s in d_hits}
        s_score_map = {i: s for i, s in s_hits}
        candidates = set(d_score_map) | set(s_score_map)
        
        merged = []
        for i in candidates:
            d_s = d_score_map.get(i, 0.0)
            s_s = s_score_map.get(i, 0.0) if sparse_applicable else 0.0
            src = []
            if i in d_score_map: src.append("dense")
            if i in s_score_map and sparse_applicable: src.append("sparse")
            
            if args.prefusion == "rrf":
                rrf_d = 1.0 / (args.rrf_k + d_rank[i]) if i in d_rank else 0.0
                rrf_s = 1.0 / (args.rrf_k + s_rank[i]) if i in s_rank and sparse_applicable else 0.0
                pf_s = rrf_d + rrf_s
            elif args.prefusion == "weighted":
                pf_s = d_norm.get(i, 0.0) * args.w_dense + (s_norm.get(i, 0.0) * args.w_sparse if sparse_applicable else 0.0)
            else:
                pf_s = d_s
                
            merged.append((i, d_s, s_s, pf_s, src))
            
        merged.sort(key=lambda x: x[3], reverse=True)
        
        results = []
        for rank, (i, d_s, s_s, pf_s, src) in enumerate(merged, 1):
            results.append({
                "rank": rank,
                "product_id": product_ids[i],
                "dense_score": d_s,
                "sparse_score": s_s if sparse_applicable else None,
                "prefusion_score": pf_s,
                "source": src,
            })
            
        out_records.append({
            "query_id": qid,
            "query_type": qtype,
            "run_name": "hybrid_dense_sparse_prefusion",
            "sparse_applicable": sparse_applicable,
            "results": results,
        })
        
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_records: f.write(json.dumps(r) + "\n")
    print(f"[retrieve] wrote {len(out_records)} records to {args.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
