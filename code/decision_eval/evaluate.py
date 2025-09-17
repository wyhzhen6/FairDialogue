import os
import json
import itertools
from collections import defaultdict
import argparse

def read_json_file(file_path: str):
    """
    Read a JSON file containing model outputs.
    Each record should contain:
      - "seg_id": segment identifier
      - "group": group label (e.g., "elder_male", "young_female")
      - "decision": model output (1/0 or 'yes'/'no')
    Returns a list of tuples: (seg_id, group, decision)
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        for rec in records:
            seg_id = rec.get("seg_id")
            group = rec.get("group")
            decision = rec.get("decision")
            if seg_id and group and decision is not None:
                if isinstance(decision, str):
                    decision = 1 if decision.lower() == "yes" else 0
                data.append((seg_id, group, decision))
    return data

def calculate_group_unfairness(data):
    """
    Calculate Group Unfairness Score (GUS).
    Returns:
      - pairwise_unfairness: dict with keys 'group1_vs_group2' and their avg unfairness
      - total_unfairness: average unfairness across all pairs and all segments
    """
    seg_groups = defaultdict(list)
    for seg_id, group, value in data:
        seg_groups[seg_id].append((group, value))

    pairwise_sums = defaultdict(list)
    total_pairs = set()

    for seg_id, entries in seg_groups.items():
        # compute per-group average for this segment
        group_values = defaultdict(list)
        for group, val in entries:
            group_values[group].append(val)
        group_avg = {g: sum(v)/len(v) for g, v in group_values.items()}

        groups = list(group_avg.keys())
        if len(groups) < 2:
            continue

        # compute pairwise absolute differences
        for g1, g2 in itertools.combinations(groups, 2):
            diff = abs(group_avg[g1] - group_avg[g2])
            pair_name = f"{g1}_vs_{g2}"
            pairwise_sums[pair_name].append(diff)
            total_pairs.add(pair_name)

    # calculate average unfairness for each pair
    pairwise_unfairness = {pair: sum(vals)/len(vals) for pair, vals in pairwise_sums.items()}

    # calculate overall average unfairness across all pairs and all segments
    total_unfairness = sum([val for vals in pairwise_sums.values() for val in vals]) / sum([len(vals) for vals in pairwise_sums.values()]) if pairwise_sums else 0.0

    return pairwise_unfairness, total_unfairness

def save_results(output_path: str, pairwise_unfairness: dict, total_unfairness: float):
    """
    Save results to JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        "pairwise_unfairness": pairwise_unfairness,
        "total_unfairness": total_unfairness
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute Group Unfairness Score (GUS) from a JSON file")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save results JSON file")
    args = parser.parse_args()

    data = read_json_file(args.input)
    pairwise_unfairness, total_unfairness = calculate_group_unfairness(data)
    save_results(args.output, pairwise_unfairness, total_unfairness)

if __name__ == "__main__":
    main()
