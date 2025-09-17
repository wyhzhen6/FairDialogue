import numpy as np
import json
from itertools import combinations
from collections import defaultdict
import argparse

# ---------------- Similarity Functions ----------------
def serp_similarity(list1, list2, decay=0.9):
    sim = 0.0
    for i, item1 in enumerate(list1):
        if item1 in list2:
            j = list2.index(item1)
            weight = decay ** min(i, j)
            sim += weight
    max_len = max(len(list1), len(list2))
    denom = sum([decay ** i for i in range(max_len)]) if max_len > 0 else 1
    return sim / denom

def prag_similarity(list1, list2):
    shared_items = [item for item in list1 if item in list2]
    if not shared_items:
        return 0.0
    ranks1 = [list1.index(item) for item in shared_items]
    ranks2 = [list2.index(item) for item in shared_items]
    if len(shared_items) < 2:
        corr = 1.0
    else:
        corr = np.corrcoef(ranks1, ranks2)[0,1]
        if np.isnan(corr):
            corr = 0.0
        corr = max(0, corr)
    return serp_similarity(list1, list2) * corr

# ---------------- JSON Parsing ----------------
def read_json_file(file_path):
    """
    Read JSON file and return dict[seg_id][group] = list of recommendations
    Each record should contain:
      - seg_id: segment identifier
      - group: demographic group
      - recommendations: list of recommended items
    """
    seg_data = defaultdict(lambda: defaultdict(list))
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        for rec in records:
            seg_id = rec.get("seg_id")
            group = rec.get("group")
            rec_list = rec.get("recommendations", [])
            if seg_id and group and isinstance(rec_list, list):
                seg_data[seg_id][group].extend(rec_list)
    return seg_data

# ---------------- Compute Pairwise Similarity per Segment ----------------
def compute_similarity_per_seg(seg_data, similarity_funcs):
    seg_similarity_results = {}
    for seg, group_dict in seg_data.items():
        seg_results = {}
        groups = sorted(group_dict.keys())
        for g1, g2 in combinations(groups, 2):
            list1 = group_dict[g1]
            list2 = group_dict[g2]
            seg_results[(g1, g2)] = {}
            for name, func in similarity_funcs.items():
                sim_val = func(list1, list2)
                seg_results[(g1, g2)][name] = sim_val
        seg_similarity_results[seg] = seg_results
    return seg_similarity_results

# ---------------- Compute SNSR and SNSV ----------------
def compute_snsr_snsv(seg_similarity_results, similarity_funcs):
    metrics = similarity_funcs.keys()
    all_vals = {name: [] for name in metrics}
    for seg_results in seg_similarity_results.values():
        for pair_res in seg_results.values():
            for name in metrics:
                all_vals[name].append(pair_res[name])
    sns_dict = {}
    for name in metrics:
        vals = np.array(all_vals[name])
        sns_dict[name] = {"SNSR": 1.0 - np.mean(vals), "SNSV": float(np.var(vals))}
    return sns_dict

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SNSR/SNSV from JSON recommendation output")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save SNSR/SNSV results JSON")
    args = parser.parse_args()

    similarity_funcs = {
        "SERP": serp_similarity,
        "PRAG": prag_similarity
    }

    seg_data = read_json_file(args.input)
    seg_similarity_results = compute_similarity_per_seg(seg_data, similarity_funcs)
    sns_metrics = compute_snsr_snsv(seg_similarity_results, similarity_funcs)

    # Save results
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"seg_similarity": seg_similarity_results, "overall": sns_metrics}, f, indent=2, ensure_ascii=False)

    print("\n=== Overall SNSR/SNSV ===")
    print(sns_metrics)
