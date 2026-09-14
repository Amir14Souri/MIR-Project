import argparse
import json
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--target_source", type=str, default="existing_or_fallback")
    parser.add_argument("--catalog", type=str, default="data/catalog_subset.parquet")
    parser.add_argument("--queries", type=str, default="data/queries.jsonl")
    parser.add_argument("--reranked", type=str, default="outputs/reranked_results.jsonl")
    parser.add_argument("--top_k", type=int, default=8)
    return parser.parse_args()

def build_prompt(query, results, catalog):
    system = (
        "You are a product search assistant. Return ONLY valid JSON matching the exact requested schema. "
        "Use only the product metadata provided. Do not invent price, availability, reviews, materials, or stock status. "
        "If a field is unknown, mark it as unknown or leave it out."
    )
    prompt = "User Query:\n"
    if query.get("query_text"): prompt += f"Text: {query['query_text']}\n"
    if query.get("query_image_path"): prompt += f"Image Provided: Yes (reference: {query['query_image_path']})\n"
    prompt += "\nTop Retrieved Products:\n"
    for r in results:
        pid = r["product_id"]
        meta = catalog.get(pid, {})
        prompt += f"- Product ID: {pid}\n"
        prompt += f"  Title: {meta.get('title', '')}\n"
        prompt += f"  Type: {meta.get('product_type', '')}\n"
        prompt += f"  Category: {meta.get('category_path', '')}\n"
        prompt += f"  Color: {meta.get('color', '')}\n"
        prompt += f"  Material: {meta.get('material', '')}\n"
        prompt += f"  Features: {meta.get('bullet_points', '')[:200]}...\n\n"
    prompt += (
        "Respond with a JSON object containing EXACTLY these keys:\n"
        "- \"query_id\": the query ID\n"
        "- \"interpreted_need\": { \"category\": \"...\", \"use_case\": \"...\", \"positive_preferences\": [], \"negative_constraints\": [], \"visual_preferences\": [], \"uncertain_fields\": [] }\n"
        "- \"product_judgements\": [ { \"product_id\": \"...\", \"role\": \"exact\" or \"substitute\" or \"irrelevant\", \"evidence\": [], \"constraint_violations\": [], \"reason\": \"...\" } ]\n"
        "- \"decision\": one of [\"recommend_exact\", \"recommend_exact_with_warning\", \"recommend_substitute\", \"no_good_match\", \"ask_clarification\"]\n"
        "- \"customer_response\": \"...\"\n"
        "Return ONLY the raw JSON string with no markdown blocks (e.g. do not wrap in ```json)."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

def main():
    args = parse_args()
    catalog = pd.read_parquet(args.catalog).set_index("product_id").to_dict("index")
    queries = {}
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            queries[d["query_id"]] = d
            
    reranked = {}
    with open(args.reranked, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            reranked[d["query_id"]] = d.get("results", [])[:args.top_k]
            
    sft_data = []
    with open(args.answers, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ans = json.loads(line)
            qid = ans["query_id"]
            if qid not in queries or qid not in reranked: continue
            
            messages = build_prompt(queries[qid], reranked[qid], catalog)
            messages.append({"role": "assistant", "content": json.dumps(ans)})
            sft_data.append({"messages": messages})
            
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in sft_data:
            f.write(json.dumps(d) + "\n")
    print(f"[sft] Wrote {len(sft_data)} examples to {args.out}")

if __name__ == "__main__":
    main()
