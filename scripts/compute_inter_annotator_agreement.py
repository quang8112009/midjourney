"""Calculate inter-annotator agreement (Cohen's kappa, raw agreement, disagreement breakdown)
and re-score metrics against Annotator 2 and consensus subset.
"""

from __future__ import annotations

import csv
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
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }


def compute_cohens_kappa(labels1: list[str], labels2: list[str], categories: list[str]) -> tuple[float, float]:
    assert len(labels1) == len(labels2)
    n = len(labels1)
    if n == 0:
        return 0.0, 0.0

    # Observed agreement
    po = sum(1 for l1, l2 in zip(labels1, labels2, strict=True) if l1 == l2) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        p1 = sum(1 for l1 in labels1 if l1 == cat) / n
        p2 = sum(1 for l2 in labels2 if l2 == cat) / n
        pe += p1 * p2

    kappa = (po - pe) / max(1.0 - pe, 1e-9) if (1.0 - pe) > 0 else 1.0
    return round(kappa, 4), round(po * 100, 2)


def main(labels2_csv_path: Path):
    mapping_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "hidden_mapping.json"
    labels1_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "human_depth_labels.csv"

    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    # Read Annotator 1
    labels1: dict[str, str] = {}
    with open(labels1_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                labels1[row[0].strip()] = row[1].strip().lower()

    # Read Annotator 2
    labels2: dict[str, str] = {}
    with open(labels2_csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                labels2[row[0].strip()] = row[1].strip().lower()

    # Match over all samples
    all_pairs = []
    for item in mapping:
        aid = item["anonymous_id"]
        l1 = labels1.get(aid, "unanswered")
        l2 = labels2.get(aid, "unanswered")
        if l1 != "unanswered" and l2 != "unanswered":
            all_pairs.append((item, l1, l2))

    print("=" * 90)
    print(f"INTER-ANNOTATOR AGREEMENT ANALYSIS (N = {len(all_pairs)} Paired Images)")
    print("=" * 90)

    # 1. 3-Way Agreement (Yes / No / Can't Tell)
    l1_3way = [l1 for _, l1, l2 in all_pairs]
    l2_3way = [l2 for _, l1, l2 in all_pairs]
    kappa_3way, po_3way = compute_cohens_kappa(l1_3way, l2_3way, ["yes", "no", "cant_tell"])

    print("\n1. THREE-WAY LABEL AGREEMENT (Yes / No / Can't Tell):")
    print(f"  Raw Percent Agreement: {po_3way:.2f}% ({sum(1 for l1, l2 in zip(l1_3way, l2_3way, strict=True) if l1 == l2)} / {len(all_pairs)})")
    print(f"  Cohen's Kappa (kappa):  {kappa_3way:.4f}")

    # 2. Binary Agreement (images where BOTH marked Yes or No)
    binary_pairs = [(it, l1 == "yes", l2 == "yes") for it, l1, l2 in all_pairs if l1 in ("yes", "no") and l2 in ("yes", "no")]
    l1_bin = [l1 for _, l1, l2 in binary_pairs]
    l2_bin = [l2 for _, l1, l2 in binary_pairs]
    kappa_bin, po_bin = compute_cohens_kappa(
        ["yes" if b else "no" for b in l1_bin],
        ["yes" if b else "no" for b in l2_bin],
        ["yes", "no"],
    )

    print(f"\n2. BINARY LABEL AGREEMENT (Both marked evaluable, N = {len(binary_pairs)}):")
    print(f"  Raw Percent Agreement: {po_bin:.2f}% ({sum(1 for b1, b2 in zip(l1_bin, l2_bin, strict=True) if b1 == b2)} / {len(binary_pairs)})")
    print(f"  Cohen's Kappa (kappa):  {kappa_bin:.4f}")


    # 3. Can't Tell Agreement
    both_cant_tell = sum(1 for _, l1, l2 in all_pairs if l1 == "cant_tell" and l2 == "cant_tell")
    l1_only_cant = sum(1 for _, l1, l2 in all_pairs if l1 == "cant_tell" and l2 != "cant_tell")
    l2_only_cant = sum(1 for _, l1, l2 in all_pairs if l1 != "cant_tell" and l2 == "cant_tell")
    neither_cant = sum(1 for _, l1, l2 in all_pairs if l1 != "cant_tell" and l2 != "cant_tell")

    print("\n3. 'CAN'T TELL' UNEVALUABILITY AGREEMENT:")
    print(f"  Both Agreed Unevaluable: {both_cant_tell}")
    print(f"  Annotator 1 Only:        {l1_only_cant}")
    print(f"  Annotator 2 Only:        {l2_only_cant}")
    print(f"  Both Agreed Evaluable:   {neither_cant}")

    # 4. Metric Re-Scoring against Annotator 2 alone
    eval_a2 = [(it, l2 == "yes") for it, _, l2 in all_pairs if l2 in ("yes", "no")]
    y2_true = [h for _, h in eval_a2]
    y2_pred_2d = [it["verdict_2d_predicate"] for it, _ in eval_a2]
    y2_pred_3d = [it["verdict_depth_anything_v2"] for it, _ in eval_a2]

    m_2d_a2 = compute_binary_metrics(y2_true, y2_pred_2d)
    m_3d_a2 = compute_binary_metrics(y2_true, y2_pred_3d)

    maj_a2 = (sum(y2_true) / len(y2_true)) * 100.0

    print("\n" + "=" * 90)
    print("4. METRIC ACCURACY RE-SCORED AGAINST ANNOTATOR 2 (N = " + str(len(eval_a2)) + ")")
    print("=" * 90)
    print(f"Majority Baseline (A2): {maj_a2:.2f}%")
    print(f"2D Ground-Plane Predicate: Acc = {m_2d_a2['accuracy']:.2f}%, Prec = {m_2d_a2['precision']:.2f}%, Rec = {m_2d_a2['recall']:.2f}%, F1 = {m_2d_a2['f1']:.2f}% (FP={m_2d_a2['fp']}, FN={m_2d_a2['fn']})")
    print(f"Depth Anything V2 (3D):    Acc = {m_3d_a2['accuracy']:.2f}%, Prec = {m_3d_a2['precision']:.2f}%, Rec = {m_3d_a2['recall']:.2f}%, F1 = {m_3d_a2['f1']:.2f}% (FP={m_3d_a2['fp']}, FN={m_3d_a2['fn']})")

    # 5. Metric Re-Scoring against Consensus Subset (Both agree on Yes or No)
    consensus_pairs = [(it, l1 == "yes") for it, l1, l2 in all_pairs if l1 in ("yes", "no") and l1 == l2]
    yc_true = [h for _, h in consensus_pairs]
    yc_pred_2d = [it["verdict_2d_predicate"] for it, _ in consensus_pairs]
    yc_pred_3d = [it["verdict_depth_anything_v2"] for it, _ in consensus_pairs]

    m_2d_cons = compute_binary_metrics(yc_true, yc_pred_2d)
    m_3d_cons = compute_binary_metrics(yc_true, yc_pred_3d)
    maj_cons = (sum(yc_true) / len(yc_true)) * 100.0 if yc_true else 0.0

    print("\n" + "=" * 90)
    print("5. METRIC ACCURACY RE-SCORED AGAINST CONSENSUS SUBSET (N = " + str(len(consensus_pairs)) + ")")
    print("=" * 90)
    print(f"Majority Baseline (Consensus): {maj_cons:.2f}%")
    print(f"2D Ground-Plane Predicate: Acc = {m_2d_cons['accuracy']:.2f}%, Prec = {m_2d_cons['precision']:.2f}%, Rec = {m_2d_cons['recall']:.2f}%, F1 = {m_2d_cons['f1']:.2f}% (FP={m_2d_cons['fp']}, FN={m_2d_cons['fn']})")
    print(f"Depth Anything V2 (3D):    Acc = {m_3d_cons['accuracy']:.2f}%, Prec = {m_3d_cons['precision']:.2f}%, Rec = {m_3d_cons['recall']:.2f}%, F1 = {m_3d_cons['f1']:.2f}% (FP={m_3d_cons['fp']}, FN={m_3d_cons['fn']})")

    out_payload = {
        "num_paired_samples": len(all_pairs),
        "three_way_agreement": {"kappa": kappa_3way, "percent_agreement": po_3way},
        "binary_agreement": {"kappa": kappa_bin, "percent_agreement": po_bin, "num_binary": len(binary_pairs)},
        "cant_tell_agreement": {
            "both_cant_tell": both_cant_tell,
            "annotator1_only": l1_only_cant,
            "annotator2_only": l2_only_cant,
            "both_evaluable": neither_cant,
        },
        "metrics_vs_annotator2": {"2d": m_2d_a2, "3d": m_3d_a2, "majority_baseline": maj_a2},
        "metrics_vs_consensus": {"2d": m_2d_cons, "3d": m_3d_cons, "majority_baseline": maj_cons, "num_consensus": len(consensus_pairs)},
    }

    out_file = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "inter_annotator_agreement_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[+] Agreement results saved to: {out_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_p = Path(sys.argv[1])
    else:
        csv_p = ROOT_DIR / "benchmarks" / "blind_depth_annotator_2" / "human_depth_labels_annotator_2.csv"

    if not csv_p.exists():
        print(f"[-] CSV file '{csv_p}' not found. Please provide path to completed Annotator 2 CSV.")
        sys.exit(1)

    main(csv_p)
