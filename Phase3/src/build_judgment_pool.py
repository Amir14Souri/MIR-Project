import argparse
import json
import csv
import os
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Build and finalize judgment pools for Phase 3.")
    parser.add_argument("--pool_out", type=str, default="data/qrels_pool.csv", help="Output CSV path for the judgment pool.")
    parser.add_argument("--finalize", type=str, help="Input CSV path containing annotated judgments.")
    parser.add_argument("--qrels_out", type=str, default="data/qrels.jsonl", help="Output JSONL path for the finalized qrels.")
    parser.add_argument("--retrieval_run", type=str, default="outputs/retrieval_results.jsonl")
    parser.add_argument("--reranked_run", type=str, default="outputs/reranked_results.jsonl")
    parser.add_argument("--queries", type=str, default="data/queries.jsonl")
    parser.add_argument("--catalog", type=str, default="data/catalog_subset.parquet")
    parser.add_argument("--top_k", type=int, default=10, help="Number of items to pool from each strategy.")
    return parser.parse_args()

def build_pool(retrieval_run, reranked_run, pool_out, top_k, queries_path, catalog_path):
    # Load context for human annotators
    queries_text = {}
    queries_img = {}
    if os.path.exists(queries_path):
        with open(queries_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                q = json.loads(line)
                queries_text[q["query_id"]] = q.get("query_text", "")
                queries_img[q["query_id"]] = q.get("query_image_path", "")

    catalog_text = {}
    catalog_img = {}
    if os.path.exists(catalog_path):
        df = pd.read_parquet(catalog_path)
        for row in df.itertuples():
            catalog_text[row.product_id] = getattr(row, 'product_text', getattr(row, 'title', ''))
            catalog_img[row.product_id] = getattr(row, 'image_path', '')

    pool = {}
    # Process Hybrid, Dense, and Sparse
    if os.path.exists(retrieval_run):
        with open(retrieval_run, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                qid = data["query_id"]
                if qid not in pool: pool[qid] = set()
                
                results = data.get("results", [])
                for r in results[:top_k]: pool[qid].add(r["product_id"])
                    
                dense_sorted = sorted([r for r in results if r.get("dense_score") is not None], key=lambda x: x["dense_score"], reverse=True)
                for r in dense_sorted[:top_k]: pool[qid].add(r["product_id"])
                    
                sparse_sorted = sorted([r for r in results if r.get("sparse_score") is not None], key=lambda x: x["sparse_score"], reverse=True)
                for r in sparse_sorted[:top_k]: pool[qid].add(r["product_id"])

    # Process CE
    if os.path.exists(reranked_run):
        with open(reranked_run, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                qid = data["query_id"]
                if qid not in pool: pool[qid] = set()
                
                for r in data.get("results", [])[:top_k]:
                    pool[qid].add(r["product_id"])

    # Write to CSV with rich context columns
    os.makedirs(os.path.dirname(pool_out), exist_ok=True)
    with open(pool_out, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "product_id", "relevance", "reason", 
            "query_text", "query_image_path", "product_text", "product_image_path"
        ])
        for qid in sorted(pool.keys()):
            for pid in sorted(list(pool[qid])):
                q_text = queries_text.get(qid, "")
                q_img = queries_img.get(qid, "")
                p_text = catalog_text.get(pid, "")
                p_img = catalog_img.get(pid, "")
                writer.writerow([qid, pid, "", "", q_text, q_img, p_text, p_img])
                
    print(f"Judgment pool written to {pool_out}.")
    print(f"Added query/product text and image paths.")

def finalize_pool(pool_in, qrels_out):
    if not os.path.exists(pool_in):
        print(f"Error: Annotated pool {pool_in} not found.")
        return
        
    records = []
    with open(pool_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("relevance", "").strip(): continue
            
            records.append({
                "query_id": row["query_id"],
                "product_id": row["product_id"],
                "relevance": int(row["relevance"]),
                "reason": row["reason"]
            })
            
    os.makedirs(os.path.dirname(qrels_out), exist_ok=True)
    with open(qrels_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully finalized {len(records)} judgments to {qrels_out}")

if __name__ == "__main__":
    args = parse_args()
    if args.finalize:
        finalize_pool(args.finalize, args.qrels_out)
    else:
        build_pool(args.retrieval_run, args.reranked_run, args.pool_out, args.top_k, args.queries, args.catalog)
