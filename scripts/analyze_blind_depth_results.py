"""Analyze human depth validation labels against the hidden mapping.

Calculates:
- Number of 'Can't tell' / unidentifiable cases (reported and excluded from binary agreement).
- Accuracy, Precision, Recall, F1 for 2D Ground-Plane Predicate.
- Accuracy, Precision, Recall, F1 for Depth Anything V2 Monocular Depth.
- Overall and split by condition (OFF 0.00 vs ON 6.00).
- Explicit conclusion on whether Case Study A's framing holds.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt and yp)
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if not yt and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt and not yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if not yt and not yp)

    total = len(y_true)
    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1e-9) if (tp + fp) > 0 else 0.0
    rec = tp / max(tp + fn, 1e-9) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-9) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def analyze(labels_csv_path: Path) -> dict:
    mapping_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "hidden_mapping.json"
    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    # Read human CSV
    human_labels: dict[str, str] = {}
    with open(labels_csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        _header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                human_labels[row[0].strip()] = row[1].strip().lower()


    cant_tell_count = 0
    unanswered_count = 0
    valid_pairs = []

    for item in mapping:
        aid = item["anonymous_id"]
        label = human_labels.get(aid, "unanswered")
        if label == "cant_tell":
            cant_tell_count += 1
        elif label in ("yes", "no"):
            valid_pairs.append((item, label == "yes"))
        else:
            unanswered_count += 1

    print("=" * 90)
    print("HUMAN DEPTH VALIDATION ANALYSIS (CASE STUDY A)")
    print("=" * 90)
    print(f"Total Samples:      {len(mapping)}")
    print(f"Human Yes / No:     {len(valid_pairs)}")
    print(f"Can't Tell:         {cant_tell_count} (excluded from binary precision/recall)")
    print(f"Unanswered:         {unanswered_count}")

    if not valid_pairs:
        print("[-] No valid Yes/No labels found to analyze.")
        return {}

    y_true_all = [h for _, h in valid_pairs]
    y_2d_all = [item["verdict_2d_predicate"] for item, _ in valid_pairs]
    y_3d_all = [item["verdict_depth_anything_v2"] for item, _ in valid_pairs]

    m_2d_all = compute_metrics(y_true_all, y_2d_all)
    m_3d_all = compute_metrics(y_true_all, y_3d_all)

    print("\n--- OVERALL AGREEMENT AGAINST HUMAN GROUND TRUTH ---")
    print(
        f"{'Metric':<30} | {'2D Ground-Plane Predicate':<26} | {'Depth Anything V2 (3D)':<26}"
    )
    print("-" * 90)
    print(
        f"{'Accuracy':<30} | {m_2d_all['accuracy']*100:6.2f}% ({m_2d_all['tp']+m_2d_all['tn']}/{m_2d_all['total']}){'':<9} | {m_3d_all['accuracy']*100:6.2f}% ({m_3d_all['tp']+m_3d_all['tn']}/{m_3d_all['total']})"
    )
    print(
        f"{'Precision':<30} | {m_2d_all['precision']*100:6.2f}%{'':<18} | {m_3d_all['precision']*100:6.2f}%"
    )
    print(
        f"{'Recall':<30} | {m_2d_all['recall']*100:6.2f}%{'':<18} | {m_3d_all['recall']*100:6.2f}%"
    )
    print(
        f"{'F1 Score':<30} | {m_2d_all['f1']*100:6.2f}%{'':<18} | {m_3d_all['f1']*100:6.2f}%"
    )
    print(
        f"{'False Positives (Hallucinations)':<30} | {m_2d_all['fp']:<26} | {m_3d_all['fp']:<26}"
    )
    print(
        f"{'False Negatives':<30} | {m_2d_all['fn']:<26} | {m_3d_all['fn']:<26}"
    )
    print("-" * 90)

    # Condition split
    off_pairs = [(item, h) for item, h in valid_pairs if item["condition_strength"] == 0.00]
    on_pairs = [(item, h) for item, h in valid_pairs if item["condition_strength"] == 6.00]

    m_2d_off = compute_metrics(
        [h for _, h in off_pairs], [item["verdict_2d_predicate"] for item, _ in off_pairs]
    )
    m_3d_off = compute_metrics(
        [h for _, h in off_pairs], [item["verdict_depth_anything_v2"] for item, _ in off_pairs]
    )

    m_2d_on = compute_metrics(
        [h for _, h in on_pairs], [item["verdict_2d_predicate"] for item, _ in on_pairs]
    )
    m_3d_on = compute_metrics(
        [h for _, h in on_pairs], [item["verdict_depth_anything_v2"] for item, _ in on_pairs]
    )

    print("\n--- CONDITION SPLIT ---")
    print(
        f"Condition OFF (0.00, N={len(off_pairs)}): 2D Acc = {m_2d_off['accuracy']*100:.2f}% (F1={m_2d_off['f1']*100:.2f}%) | Depth Anything V2 Acc = {m_3d_off['accuracy']*100:.2f}% (F1={m_3d_off['f1']*100:.2f}%)"
    )
    print(
        f"Condition ON  (6.00, N={len(on_pairs)}): 2D Acc = {m_2d_on['accuracy']*100:.2f}% (F1={m_2d_on['f1']*100:.2f}%) | Depth Anything V2 Acc = {m_3d_on['accuracy']*100:.2f}% (F1={m_3d_on['f1']*100:.2f}%)"
    )

    res_dict = {
        "overall": {"2d": m_2d_all, "3d": m_3d_all},
        "condition_off_0.00": {"2d": m_2d_off, "3d": m_3d_off},
        "condition_on_6.00": {"2d": m_2d_on, "3d": m_3d_on},
        "cant_tell_count": cant_tell_count,
        "unanswered_count": unanswered_count,
    }

    out_json = (
        ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "depth_human_validation_results.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, indent=2)
    print(f"\n[+] Analysis saved to: {out_json}")
    return res_dict


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
    else:
        csv_file = (
            ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "human_depth_labels.csv"
        )

    if not csv_file.exists():
        print(f"[-] CSV file '{csv_file}' not found. Please provide path to completed CSV.")
        sys.exit(1)

    analyze(csv_file)
