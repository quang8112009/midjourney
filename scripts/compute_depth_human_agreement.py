"""Compute agreement metrics between human ground-truth labels and automated depth metrics.

Calculates Accuracy, Precision, Recall, and F1 for:
1. 2D Ground-Plane Predicate
2. Depth Anything V2 Monocular Depth Estimator
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


def compute_binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
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


def main():
    secret_path = ROOT_DIR / "benchmarks" / "human_depth_validation_secret_key.json"
    labels_path = ROOT_DIR / "benchmarks" / "human_labels.json"

    if not labels_path.exists():
        print(f"[-] Missing {labels_path}. Please complete human labeling first.")
        return 1

    with open(secret_path, encoding="utf-8") as f:
        secret_data = json.load(f)

    with open(labels_path, encoding="utf-8") as f:
        human_labels = json.load(f)


    y_human = []
    y_2d = []
    y_3d = []

    for item in secret_data:
        bid = item["blind_id"]
        h_val = human_labels.get(bid)
        if h_val is None:
            print(f"[-] Warning: {bid} is unlabeled.")
            continue
        y_human.append(bool(h_val))
        y_2d.append(bool(item["metric_2d_satisfied"]))
        y_3d.append(bool(item["metric_depth_anything_satisfied"]))

    print("=" * 85)
    print(f"HUMAN VALIDATION OF DEPTH METRICS (N={len(y_human)} Blinded Samples)")
    print("=" * 85)

    m_2d = compute_binary_metrics(y_human, y_2d)
    m_3d = compute_binary_metrics(y_human, y_3d)

    print(f"\n{'Metric Evaluation':<30} | {'2D Ground-Plane Predicate':<24} | {'Depth Anything V2 (3D)':<24}")
    print("-" * 85)
    print(f"{'Accuracy vs Human (Ground Truth)':<30} | {m_2d['accuracy']*100:6.2f}% ({m_2d['tp']+m_2d['tn']}/{m_2d['total']}){'':<7} | {m_3d['accuracy']*100:6.2f}% ({m_3d['tp']+m_3d['tn']}/{m_3d['total']})")
    print(f"{'Precision':<30} | {m_2d['precision']*100:6.2f}%{'':<16} | {m_3d['precision']*100:6.2f}%")
    print(f"{'Recall':<30} | {m_2d['recall']*100:6.2f}%{'':<16} | {m_3d['recall']*100:6.2f}%")
    print(f"{'F1 Score':<30} | {m_2d['f1']*100:6.2f}%{'':<16} | {m_3d['f1']*100:6.2f}%")
    print(f"{'False Positives (Hallucinations)':<30} | {m_2d['fp']:<24} | {m_3d['fp']:<24}")
    print(f"{'False Negatives':<30} | {m_2d['fn']:<24} | {m_3d['fn']:<24}")
    print("=" * 85)

    # Breakdown by condition (OFF vs 6.00)
    cond_0_indices = [idx for idx, s in enumerate(secret_data) if s["strength"] == 0.00]
    cond_6_indices = [idx for idx, s in enumerate(secret_data) if s["strength"] == 6.00]

    y_h_0 = [y_human[i] for i in cond_0_indices]
    y_2d_0 = [y_2d[i] for i in cond_0_indices]
    y_3d_0 = [y_3d[i] for i in cond_0_indices]

    y_h_6 = [y_human[i] for i in cond_6_indices]
    y_2d_6 = [y_2d[i] for i in cond_6_indices]
    y_3d_6 = [y_3d[i] for i in cond_6_indices]

    m_2d_0 = compute_binary_metrics(y_h_0, y_2d_0)
    m_3d_0 = compute_binary_metrics(y_h_0, y_3d_0)
    m_2d_6 = compute_binary_metrics(y_h_6, y_2d_6)
    m_3d_6 = compute_binary_metrics(y_h_6, y_3d_6)

    print("\n--- SUB-GROUP BREAKDOWN BY CONDITION ---")
    print(f"OFF (0.00, N={len(y_h_0)}):")
    print(f"  2D Predicate Accuracy:       {m_2d_0['accuracy']*100:.2f}% (F1={m_2d_0['f1']*100:.2f}%)")
    print(f"  Depth Anything V2 Accuracy:  {m_3d_0['accuracy']*100:.2f}% (F1={m_3d_0['f1']*100:.2f}%)")
    print(f"Strength 6.00 (N={len(y_h_6)}):")
    print(f"  2D Predicate Accuracy:       {m_2d_6['accuracy']*100:.2f}% (F1={m_2d_6['f1']*100:.2f}%)")
    print(f"  Depth Anything V2 Accuracy:  {m_3d_6['accuracy']*100:.2f}% (F1={m_3d_6['f1']*100:.2f}%)")

    report = {
        "overall_comparison": {
            "2d_ground_plane": m_2d,
            "depth_anything_v2": m_3d,
        },
        "by_condition": {
            "off_0.00": {"2d": m_2d_0, "3d": m_3d_0},
            "strength_6.00": {"2d": m_2d_6, "3d": m_3d_6},
        },
    }

    out_file = ROOT_DIR / "benchmarks" / "depth_human_validation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Saved validation results to: {out_file}")


if __name__ == "__main__":
    main()
