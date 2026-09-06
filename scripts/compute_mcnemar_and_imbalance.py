import csv
import json
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

mapping_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "hidden_mapping.json"
labels_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "human_depth_labels.csv"

with open(mapping_path, encoding="utf-8") as f:
    mapping = json.load(f)

human_labels = {}
with open(labels_path, encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if len(row) >= 2:
            human_labels[row[0].strip()] = row[1].strip().lower()

# 1. Inspect Sample Balance vs Human 73/17 split
total_sample = len(mapping)
front_in_sample = sum(1 for m in mapping if m["relation"] == "in_front_of")
behind_in_sample = sum(1 for m in mapping if m["relation"] == "behind")
off_in_sample = sum(1 for m in mapping if m["condition_strength"] == 0.0)
on_in_sample = sum(1 for m in mapping if m["condition_strength"] == 6.0)

print("=" * 80)
print("1. SAMPLE BALANCE AUDIT")
print("=" * 80)
print(f"Total Sampled:         {total_sample}")
print(f"Relation in_front_of:  {front_in_sample} (50.0%)")
print(f"Relation behind:       {behind_in_sample} (50.0%)")
print(f"Condition OFF (0.00):  {off_in_sample} (50.0%)")
print(f"Condition ON (6.00):   {on_in_sample} (50.0%)")

# Inspect the 30 Can't-tell
cant_tell_items = [m for m in mapping if human_labels.get(m["anonymous_id"]) == "cant_tell"]
cant_front = sum(1 for m in cant_tell_items if m["relation"] == "in_front_of")
cant_behind = sum(1 for m in cant_tell_items if m["relation"] == "behind")
cant_off = sum(1 for m in cant_tell_items if m["condition_strength"] == 0.0)
cant_on = sum(1 for m in cant_tell_items if m["condition_strength"] == 6.0)

print(f"\nCan't Tell (N={len(cant_tell_items)} / 120 = {len(cant_tell_items)/120*100:.1f}%):")
print(f"  From in_front_of:    {cant_front}")
print(f"  From behind:         {cant_behind}")
print(f"  From OFF (0.00):     {cant_off}")
print(f"  From ON (6.00):      {cant_on}")

# Evaluable items
eval_items = [(m, human_labels[m["anonymous_id"]] == "yes") for m in mapping if human_labels.get(m["anonymous_id"]) in ("yes", "no")]
print(f"\nEvaluable Binary Samples (N={len(eval_items)}):")
eval_yes = sum(1 for _, y in eval_items if y)
eval_no = sum(1 for _, y in eval_items if not y)
print(f"  Human YES:           {eval_yes} ({eval_yes/len(eval_items)*100:.2f}%)")
print(f"  Human NO:            {eval_no} ({eval_no/len(eval_items)*100:.2f}%)")
print(f"  Majority Baseline:   {eval_yes/len(eval_items)*100:.2f}%")

eval_front_yes = sum(1 for m, y in eval_items if m["relation"] == "in_front_of" and y)
eval_front_no = sum(1 for m, y in eval_items if m["relation"] == "in_front_of" and not y)
eval_behind_yes = sum(1 for m, y in eval_items if m["relation"] == "behind" and y)
eval_behind_no = sum(1 for m, y in eval_items if m["relation"] == "behind" and not y)

print(f"  in_front_of subset:  {eval_front_yes} YES vs {eval_front_no} NO (Total: {eval_front_yes+eval_front_no})")
print(f"  behind subset:       {eval_behind_yes} YES vs {eval_behind_no} NO (Total: {eval_behind_yes+eval_behind_no})")

# 2. McNemar Paired Test: 2D Predicate vs Depth Anything V2 on Human Agreement
correct_2d = [item["verdict_2d_predicate"] == y_true for item, y_true in eval_items]
correct_3d = [item["verdict_depth_anything_v2"] == y_true for item, y_true in eval_items]

# Contingency table:
# b: 2D correct, 3D incorrect
# c: 2D incorrect, 3D correct
a = sum(1 for c2, c3 in zip(correct_2d, correct_3d, strict=True) if c2 and c3)
b = sum(1 for c2, c3 in zip(correct_2d, correct_3d, strict=True) if c2 and not c3)
c = sum(1 for c2, c3 in zip(correct_2d, correct_3d, strict=True) if not c2 and c3)
d = sum(1 for c2, c3 in zip(correct_2d, correct_3d, strict=True) if not c2 and not c3)


print("\n" + "=" * 80)
print("2. MCNEMAR PAIRED TEST: 2D PREDICATE vs DEPTH ANYTHING V2 ACCURACY")
print("=" * 80)
print(f"Both Correct (a):              {a}")
print(f"2D Correct & 3D Wrong (b):     {b}")
print(f"2D Wrong & 3D Correct (c):     {c}")
print(f"Both Wrong (d):                {d}")
print(f"Net Difference (c - b):        {c - b} (out of {len(eval_items)})")

# Exact binomial two-tailed p-value for McNemar
n_discordant = b + c
k_min = min(b, c)
# Two-tailed exact binomial sum for p=0.5
p_exact = 2.0 * sum(math.comb(n_discordant, i) * (0.5 ** n_discordant) for i in range(k_min + 1))
p_exact = min(1.0, p_exact)
print(f"McNemar Exact Two-Tailed p-val: p = {p_exact:.4f}")


# Check metric verdicts on the 30 "Can't Tell" images
cant_tell_2d_assigned = sum(1 for m in cant_tell_items if m["verdict_2d_predicate"] is not None)
cant_tell_3d_assigned = sum(1 for m in cant_tell_items if m["verdict_depth_anything_v2"] is not None)
print(f"\nVerdicts assigned to the {len(cant_tell_items)} unevaluable images:")
print(f"  2D Predicate assigned verdicts:      {cant_tell_2d_assigned} / {len(cant_tell_items)} ({cant_tell_2d_assigned/len(cant_tell_items)*100:.0f}%)")
print(f"  Depth Anything V2 assigned verdicts: {cant_tell_3d_assigned} / {len(cant_tell_items)} ({cant_tell_2d_assigned/len(cant_tell_items)*100:.0f}%)")
