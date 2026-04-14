import argparse

import pandas as pd

from fashion_rag.search import encode_text, load_bq_index, load_model, local_search

MIN_ITEMS_FOR_QUERY = 1


def generate_queries(metadata):
    combos = (
        metadata.groupby(["baseColour", "articleType"]).size().reset_index(name="count")
    )
    combos = combos[combos["count"] >= MIN_ITEMS_FOR_QUERY]

    queries = []
    for _, row in combos.iterrows():
        colour, atype, count = row["baseColour"], row["articleType"], row["count"]
        queries.append({
            "query": f"{colour} {atype}",
            "expected": {"baseColour": colour, "articleType": atype},
            "available": count,
        })
    return queries


def reciprocal_rank(matches):
    hits = matches.values.nonzero()[0]
    return 1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0


def evaluate_results(results, expected):
    colour_match = results["baseColour"].str.lower() == expected["baseColour"].lower()
    type_match = results["articleType"].str.lower() == expected["articleType"].lower()
    both_match = colour_match & type_match

    return {
        "num_matches": both_match.sum(),
        "rr": reciprocal_rank(both_match),
        "rr_colour": reciprocal_rank(colour_match),
        "rr_type": reciprocal_rank(type_match),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    model, processor = load_model()
    print("Loading embeddings and metadata from BQ...")
    embeddings, metadata = load_bq_index()

    queries = generate_queries(metadata)
    print(f"Generated {len(queries)} queries (colour+type combos with >=3 items)\n")

    k = args.k

    results_list = []
    for q in queries:
        query_emb = encode_text(q["query"], model, processor)
        topk = local_search(query_emb, embeddings, metadata, k=k)
        ev = evaluate_results(topk, q["expected"])
        recall_denom = min(k, q["available"])
        results_list.append({
            "query": q["query"],
            "available": q["available"],
            "rr": ev["rr"],
            "rr_colour": ev["rr_colour"],
            "rr_type": ev["rr_type"],
            "r@k": ev["num_matches"] / recall_denom,
        })

    df = pd.DataFrame(results_list).sort_values("rr", ascending=False)

    lines = []
    lines.append(f"{'Query':<30} {'Avail':>5} {f'RR@{k}':>6} {'RR_col':>6} {'RR_typ':>6} {f'R@{k}':>6}")
    lines.append("-" * 64)
    for _, r in df.iterrows():
        lines.append(
            f"{r['query']:<30} {r['available']:>5} {r['rr']:>6.2f} "
            f"{r['rr_colour']:>6.2f} {r['rr_type']:>6.2f} {r['r@k']:>6.2f}"
        )
    lines.append(f"\n{'=' * 64}")
    lines.append(f"MRR@{k}:                   {df['rr'].mean():.3f}")
    lines.append(f"MRR@{k} (colour):          {df['rr_colour'].mean():.3f}")
    lines.append(f"MRR@{k} (type):            {df['rr_type'].mean():.3f}")
    lines.append(f"Mean Recall@{k}:           {df['r@k'].mean():.3f}")

    report = "\n".join(lines)
    print(report)

    out = "eval-outputs/eval_report.txt"
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
