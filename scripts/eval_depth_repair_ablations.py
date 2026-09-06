"""Comprehensive Depth Aggregation, Threshold, Auxiliary Signal, and Abstention Ablations.

Evaluates on the exact 90 evaluable human labels from benchmarks/blind_depth_evaluation/.
All metrics compared against the 81.11% majority-class baseline.
Pure numpy implementation with zero external ML dependencies.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.eval_spatial_depth_dedicated import (
    MonocularDepthEvaluator,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator

MAJORITY_BASELINE = 81.11


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


def compute_roc_auc(y_true: list[bool], scores: list[float]) -> float:
    # Exact trapezoidal ROC AUC computation in pure numpy
    pos_scores = [s for yt, s in zip(y_true, scores, strict=True) if yt]
    neg_scores = [s for yt, s in zip(y_true, scores, strict=True) if not yt]
    if not pos_scores or not neg_scores:
        return 0.5
    # Mann-Whitney U statistic formulation for exact AUC
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    u = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                u += 1.0
            elif ps == ns:
                u += 0.5
    return u / (n_pos * n_neg)


def extract_box_depths(depth_map: np.ndarray, box: list[float], h: int, w: int, method: str) -> float | None:
    ymin, xmin, ymax, xmax = box
    y1, x1, y2, x2 = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
    y1, x1, y2, x2 = max(0, y1), max(0, x1), min(h, y2), min(w, x2)
    if y2 <= y1 or x2 <= x1:
        return None

    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if method == "mean":
        return float(np.mean(crop))
    elif method == "median":
        return float(np.median(crop))
    elif method == "p10":
        return float(np.percentile(crop, 10))
    elif method == "p25":
        return float(np.percentile(crop, 25))
    elif method == "p75":
        return float(np.percentile(crop, 75))
    elif method == "p90":
        return float(np.percentile(crop, 90))
    elif method == "eroded_25":
        dy = (y2 - y1) * 0.25 / 2.0
        dx = (x2 - x1) * 0.25 / 2.0
        ey1, ey2 = int(y1 + dy), int(y2 - dy)
        ex1, ex2 = int(x1 + dx), int(x2 - dx)
        if ey2 > ey1 and ex2 > ex1:
            return float(np.mean(depth_map[ey1:ey2, ex1:ex2]))
        return float(np.mean(crop))
    elif method == "eroded_50":
        dy = (y2 - y1) * 0.50 / 2.0
        dx = (x2 - x1) * 0.50 / 2.0
        ey1, ey2 = int(y1 + dy), int(y2 - dy)
        ex1, ex2 = int(x1 + dx), int(x2 - dx)
        if ey2 > ey1 and ex2 > ex1:
            return float(np.mean(depth_map[ey1:ey2, ex1:ex2]))
        return float(np.mean(crop))
    elif method == "center_pixel":
        cy, cx = int((y1 + y2) / 2), int((x1 + x2) / 2)
        cy = min(h - 1, max(0, cy))
        cx = min(w - 1, max(0, cx))
        return float(depth_map[cy, cx])
    elif method == "foreground_adaptive_p60":
        thresh = np.percentile(crop, 60)
        fg = crop[crop >= thresh]
        return float(np.mean(fg)) if fg.size > 0 else float(np.mean(crop))
    return float(np.mean(crop))


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.05, steps: int = 500) -> tuple[np.ndarray, float]:
    # Weighted logistic regression with gradient descent
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    # Balanced class weights
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    w_pos = (n / (2.0 * max(n_pos, 1)))
    w_neg = (n / (2.0 * max(n_neg, 1)))
    sample_weights = np.where(y == 1, w_pos, w_neg)

    for _ in range(steps):
        z = np.clip(np.dot(X, w) + b, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        error = (p - y) * sample_weights
        grad_w = np.dot(X.T, error) / n
        grad_b = np.sum(error) / n
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator_2d = StrictSpatialEvaluator(device=device)
    evaluator_3d = MonocularDepthEvaluator(device=device)

    mapping_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "hidden_mapping.json"
    labels_path = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "human_depth_labels.csv"
    img_dir = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "images"

    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    human_labels = {}
    with open(labels_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                human_labels[row[0].strip()] = row[1].strip().lower()

    # Pre-extract all samples, bounding boxes, and depth maps
    print("Computing depth maps and bounding boxes for all 120 images...")
    processed_samples = []

    for item in mapping:
        aid = item["anonymous_id"]
        h_label = human_labels.get(aid, "unanswered")
        img_path = img_dir / f"{aid}.png"
        img = Image.open(img_path).convert("RGB")

        # 2D detections
        s_det, o_det = evaluator_2d.detect_entities(img, item["subject"], item["object"])

        # Depth map from Depth Anything V2
        inputs = evaluator_3d.processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = evaluator_3d.model(**inputs)
            depth_raw = outputs.predicted_depth.unsqueeze(1)
            depth_map = (
                F.interpolate(
                    depth_raw,
                    size=(img.height, img.width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        processed_samples.append({
            "id": aid,
            "prompt": item["prompt"],
            "subject": item["subject"],
            "object": item["object"],
            "relation": item["relation"],
            "strength": item["condition_strength"],
            "seed": item["seed"],
            "human_label": h_label,
            "s_det": s_det,
            "o_det": o_det,
            "depth_map": depth_map,
            "img_h": img.height,
            "img_w": img.width,
        })

    evaluable = [s for s in processed_samples if s["human_label"] in ("yes", "no")]
    y_true = [s["human_label"] == "yes" for s in evaluable]

    print(f"\nTotal Evaluable Samples: {len(evaluable)} ({sum(y_true)} YES, {len(y_true)-sum(y_true)} NO)")
    print(f"Majority-Class Baseline: {MAJORITY_BASELINE}%\n")

    # =========================================================================
    # 2. DEPTH AGGREGATION ABLATION
    # =========================================================================
    methods = [
        ("Default (Mean)", "mean"),
        ("Median", "median"),
        ("10th Percentile (p10)", "p10"),
        ("25th Percentile (p25)", "p25"),
        ("75th Percentile (p75)", "p75"),
        ("90th Percentile (p90)", "p90"),
        ("Eroded Box 25%", "eroded_25"),
        ("Eroded Box 50%", "eroded_50"),
        ("Center Pixel", "center_pixel"),
        ("Foreground Adaptive (Top 40% p60)", "foreground_adaptive_p60"),
    ]

    ablation_results = {}
    print("=" * 105)
    print("2. DEPTH AGGREGATION ABLATIONS (ON 90 EVALUABLE HUMAN LABELS)")
    print("=" * 105)
    print(f"{'Aggregation Method':<34} | {'Accuracy':<10} | {'Majority Baseline':<18} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'TP/FP/FN/TN'}")
    print("-" * 105)

    for label, m_key in methods:
        y_pred = []
        for s in evaluable:
            if s["s_det"] is None or s["o_det"] is None:
                y_pred.append(False)
                continue
            sd = extract_box_depths(s["depth_map"], s["s_det"]["box"], s["img_h"], s["img_w"], m_key)
            od = extract_box_depths(s["depth_map"], s["o_det"]["box"], s["img_h"], s["img_w"], m_key)
            if sd is None or od is None:
                y_pred.append(False)
                continue

            if s["relation"] == "in_front_of":
                sat = (sd >= od + 0.05)
            else:
                sat = (sd <= od - 0.05)
            y_pred.append(bool(sat))

        res = compute_metrics(y_true, y_pred)
        ablation_results[m_key] = res
        print(f"{label:<34} | {res['accuracy']:5.2f}%    | {MAJORITY_BASELINE:5.2f}%            | {res['precision']:5.2f}%    | {res['recall']:5.2f}%    | {res['f1']:5.2f}%  | {res['tp']}/{res['fp']}/{res['fn']}/{res['tn']}")

    # Fallback to 2D Ground-Plane Predicate for comparison
    y_pred_2d = []
    for s in evaluable:
        sat_2d, _ = evaluator_2d.check_relation(s["s_det"], s["o_det"], s["relation"])
        y_pred_2d.append(bool(sat_2d))
    res_2d = compute_metrics(y_true, y_pred_2d)
    print(f"{'2D Ground-Plane Predicate':<34} | {res_2d['accuracy']:5.2f}%    | {MAJORITY_BASELINE:5.2f}%            | {res_2d['precision']:5.2f}%    | {res_2d['recall']:5.2f}%    | {res_2d['f1']:5.2f}%  | {res_2d['tp']}/{res_2d['fp']}/{res_2d['fn']}/{res_2d['tn']}")

    # =========================================================================
    # 3. THRESHOLD ANALYSIS & ROC / AUC
    # =========================================================================
    print("\n" + "=" * 105)
    print("3. THRESHOLD ANALYSIS & ROC / AUC (SWEEPING EPSILON THRESHOLD)")
    print("=" * 105)

    continuous_scores = []
    for s in evaluable:
        if s["s_det"] is None or s["o_det"] is None:
            continuous_scores.append(-999.0)
            continue
        sd = extract_box_depths(s["depth_map"], s["s_det"]["box"], s["img_h"], s["img_w"], "mean")
        od = extract_box_depths(s["depth_map"], s["o_det"]["box"], s["img_h"], s["img_w"], "mean")
        if sd is None or od is None:
            continuous_scores.append(-999.0)
            continue
        if s["relation"] == "in_front_of":
            gap = sd - od
        else:
            gap = od - sd
        continuous_scores.append(float(gap))

    auc_all = compute_roc_auc(y_true, continuous_scores)
    detected_mask = [sc != -999.0 for sc in continuous_scores]
    y_true_det = [yt for yt, m in zip(y_true, detected_mask, strict=True) if m]
    scores_det = [sc for sc, m in zip(continuous_scores, detected_mask, strict=True) if m]
    auc_det = compute_roc_auc(y_true_det, scores_det)

    print(f"ROC AUC (All 90 samples, missing detection=-999):   AUC = {auc_all:.4f}")
    print(f"ROC AUC (Detected subset only, N={len(scores_det)}):         AUC = {auc_det:.4f}")

    thresholds = [-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    print(f"\n{'Threshold (tau)':<16} | {'Accuracy':<10} | {'Majority Baseline':<18} | {'Precision':<10} | {'Recall':<10} | {'F1':<8} | {'TP/FP/FN/TN'}")
    print("-" * 105)
    best_thresh_acc = 0.0
    best_thresh = 0.0
    for tau in thresholds:
        yp = [sc >= tau for sc in continuous_scores]
        res_t = compute_metrics(y_true, yp)
        if res_t['accuracy'] > best_thresh_acc:
            best_thresh_acc = res_t['accuracy']
            best_thresh = tau
        print(f"tau = {tau:+5.2f}{'':<6} | {res_t['accuracy']:5.2f}%    | {MAJORITY_BASELINE:5.2f}%            | {res_t['precision']:5.2f}%    | {res_t['recall']:5.2f}%    | {res_t['f1']:5.2f}%  | {res_t['tp']}/{res_t['fp']}/{res_t['fn']}/{res_t['tn']}")

    print(f"\nPeak accuracy achieved via threshold sweep: {best_thresh_acc:.2f}% at tau = {best_thresh:+.2f} (vs {MAJORITY_BASELINE:.2f}% majority baseline)")

    # =========================================================================
    # 4. AUXILIARY SIGNALS: OCCLUSION, RELATIVE SCALE & LOGISTIC REGRESSION
    # =========================================================================
    print("\n" + "=" * 105)
    print("4. AUXILIARY SIGNALS & MULTI-MODAL MODELING (50/50 TRAIN / TEST SPLIT)")
    print("=" * 105)

    y_pred_occlusion = []
    overlap_count = 0
    for s in evaluable:
        if s["s_det"] is None or s["o_det"] is None:
            y_pred_occlusion.append(False)
            continue
        s_box = s["s_det"]["box"]
        o_box = s["o_det"]["box"]
        iy1, ix1 = max(s_box[0], o_box[0]), max(s_box[1], o_box[1])
        iy2, ix2 = min(s_box[2], o_box[2]), min(s_box[3], o_box[3])
        if iy2 > iy1 and ix2 > ix1:
            overlap_count += 1
            inter_depth = s["depth_map"][int(iy1*s["img_h"]):int(iy2*s["img_h"]), int(ix1*s["img_w"]):int(ix2*s["img_w"])]
            s_crop = s["depth_map"][int(s_box[0]*s["img_h"]):int(s_box[2]*s["img_h"]), int(s_box[1]*s["img_w"]):int(s_box[3]*s["img_w"])]
            o_crop = s["depth_map"][int(o_box[0]*s["img_h"]):int(o_box[2]*s["img_h"]), int(o_box[1]*s["img_w"]):int(o_box[3]*s["img_w"])]
            sd_diff = abs(np.mean(inter_depth) - np.mean(s_crop))
            od_diff = abs(np.mean(inter_depth) - np.mean(o_crop))
            s_is_front = (sd_diff < od_diff)
            if s["relation"] == "in_front_of":
                y_pred_occlusion.append(s_is_front)
            else:
                y_pred_occlusion.append(not s_is_front)
        else:
            sd = extract_box_depths(s["depth_map"], s_box, s["img_h"], s["img_w"], "mean")
            od = extract_box_depths(s["depth_map"], o_box, s["img_h"], s["img_w"], "mean")
            if s["relation"] == "in_front_of":
                y_pred_occlusion.append(bool(sd >= od + 0.05))
            else:
                y_pred_occlusion.append(bool(sd <= od - 0.05))

    res_occ = compute_metrics(y_true, y_pred_occlusion)
    print(f"Bounding box overlap occurs in {overlap_count} / {len(evaluable)} ({overlap_count/len(evaluable)*100:.1f}%) of evaluable depth images.")
    print(f"Occlusion-Aware Depth Rule: Accuracy = {res_occ['accuracy']:.2f}%, Precision = {res_occ['precision']:.2f}%, Recall = {res_occ['recall']:.2f}%, F1 = {res_occ['f1']:.2f}% (vs {MAJORITY_BASELINE:.2f}% baseline)")

    # 4b. Logistic Regression (50/50 Train / Test Split)
    X_features = []
    for s in evaluable:
        if s["s_det"] is None or s["o_det"] is None:
            X_features.append([-5.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            continue
        sd = extract_box_depths(s["depth_map"], s["s_det"]["box"], s["img_h"], s["img_w"], "mean")
        od = extract_box_depths(s["depth_map"], s["o_det"]["box"], s["img_h"], s["img_w"], "mean")
        gap = (sd - od) if s["relation"] == "in_front_of" else (od - sd)
        s_box = s["s_det"]["box"]
        o_box = s["o_det"]["box"]
        s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
        o_area = (o_box[2] - o_box[0]) * (o_box[3] - o_box[1])
        area_ratio = s_area / max(o_area, 1e-6)
        
        iy1, ix1 = max(s_box[0], o_box[0]), max(s_box[1], o_box[1])
        iy2, ix2 = min(s_box[2], o_box[2]), min(s_box[3], o_box[3])
        has_overlap = 1.0 if (iy2 > iy1 and ix2 > ix1) else 0.0
        iou = ((iy2 - iy1) * (ix2 - ix1)) / max(s_area + o_area - (iy2 - iy1) * (ix2 - ix1), 1e-6) if has_overlap else 0.0

        X_features.append([
            float(gap),
            float(iou),
            float(min(10.0, area_ratio)),
            float(s["s_det"]["score"]),
            float(s["o_det"]["score"]),
            float(has_overlap),
        ])

    X = np.array(X_features)
    y = np.array(y_true, dtype=int)

    # Deterministic stratified 50/50 split
    rng = random.Random(42)
    pos_idx = [i for i, yt in enumerate(y) if yt == 1]
    neg_idx = [i for i, yt in enumerate(y) if yt == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    train_idx = pos_idx[:len(pos_idx)//2] + neg_idx[:len(neg_idx)//2]
    test_idx = pos_idx[len(pos_idx)//2:] + neg_idx[len(neg_idx)//2:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize features by train stats
    mean_feat = np.mean(X_train, axis=0)
    std_feat = np.std(X_train, axis=0) + 1e-6
    X_train_norm = (X_train - mean_feat) / std_feat
    X_test_norm = (X_test - mean_feat) / std_feat

    w, b = fit_logistic_regression(X_train_norm, y_train, lr=0.1, steps=1000)

    train_preds = [(1.0 / (1.0 + np.exp(-np.clip(np.dot(x, w) + b, -20.0, 20.0)))) >= 0.5 for x in X_train_norm]
    test_preds = [(1.0 / (1.0 + np.exp(-np.clip(np.dot(x, w) + b, -20.0, 20.0)))) >= 0.5 for x in X_test_norm]

    m_train = compute_metrics(y_train.tolist(), train_preds)
    m_test = compute_metrics(y_test.tolist(), test_preds)

    print(f"\nLogistic Regression (50/50 Stratified Split, Train N={len(y_train)}, Test N={len(y_test)}):")
    print(f"  Train Split: Accuracy = {m_train['accuracy']:.2f}% | Precision = {m_train['precision']:.2f}% | Recall = {m_train['recall']:.2f}% | F1 = {m_train['f1']:.2f}%")
    print(f"  Test Split:  Accuracy = {m_test['accuracy']:.2f}% | Precision = {m_test['precision']:.2f}% | Recall = {m_test['recall']:.2f}% | F1 = {m_test['f1']:.2f}% (vs {MAJORITY_BASELINE:.2f}% baseline)")

    # =========================================================================
    # 5. ABSTENTION ANALYSIS (PREDICTING THE 30 CAN'T-TELL SCENES)
    # =========================================================================
    print("\n" + "=" * 105)
    print("5. ABSTENTION ANALYSIS (PREDICTING THE 30 CAN'T-TELL / UNEVALUABLE SCENES)")
    print("=" * 105)

    y_cant_tell_true = [s["human_label"] == "cant_tell" for s in processed_samples]
    abstention_thresholds = [0.08, 0.12, 0.15, 0.20, 0.25, 0.30]

    print(f"Total Images: {len(processed_samples)} (Can't Tell = {sum(y_cant_tell_true)} / 120 = 25.00%)")
    print(f"\n{'Abstention Rule (Min Confidence < tau)':<42} | {'Abstain Acc':<12} | {'Precision':<12} | {'Recall':<12} | {'F1':<8} | {'Abstained Count'}")
    print("-" * 105)

    for thresh in abstention_thresholds:
        y_abstain_pred = []
        for s in processed_samples:
            s_sc = float(s["s_det"]["score"]) if s["s_det"] else 0.0
            o_sc = float(s["o_det"]["score"]) if s["o_det"] else 0.0
            should_abstain = (s_sc < thresh or o_sc < thresh)
            y_abstain_pred.append(should_abstain)

        res_ab = compute_metrics(y_cant_tell_true, y_abstain_pred)
        print(f"tau = {thresh:.2f}{'':<32} | {res_ab['accuracy']:5.2f}%      | {res_ab['precision']:5.2f}%      | {res_ab['recall']:5.2f}%      | {res_ab['f1']:5.2f}%  | {sum(y_abstain_pred)} / 120")

    # Save complete ablation report
    out_payload = {
        "majority_class_baseline": MAJORITY_BASELINE,
        "evaluable_samples_count": len(evaluable),
        "cant_tell_samples_count": sum(y_cant_tell_true),
        "depth_aggregation_ablations": ablation_results,
        "threshold_analysis": {
            "roc_auc_all": round(float(auc_all), 4),
            "roc_auc_detected_only": round(float(auc_det), 4),
            "best_threshold": best_thresh,
            "best_threshold_accuracy": best_thresh_acc,
        },
        "auxiliary_signals": {
            "occlusion_rule": res_occ,
            "logistic_regression_train": m_train,
            "logistic_regression_test": m_test,
        },
    }

    out_file = ROOT_DIR / "benchmarks" / "blind_depth_evaluation" / "depth_repair_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\n[+] Master depth repair ablation results saved to: {out_file}")


if __name__ == "__main__":
    main()
