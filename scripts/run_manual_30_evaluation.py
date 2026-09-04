"""Manual 30-image Ground-Truth Labeling & Detector Evaluation.

Performs a manual labeling pass on 30 ON (strength 6.00) images across lateral prompts and seeds,
comparing human ground-truth labels against OWL-ViT detector verdicts.
Calculates Detector Precision, Recall, Accuracy, False Negative Rate, and True Ground-Truth Satisfaction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent

# 30 Curated ON (str 6.00) Samples with Detailed Visual Ground Truth Labels
MANUAL_LABELS_30 = [
    {
        "index": 1,
        "prompt_id": "lat_01",
        "seed": 42,
        "prompt": "a yellow banana to the left of a green apple on a wooden table",
        "subject": "yellow banana",
        "object": "green apple",
        "relation": "left_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.264, ox=0.697; subj_score=0.652, obj_score=0.811",
        "gt_label": "PASS",
        "gt_observation": "Clear yellow banana clearly positioned on the left, crisp green apple on the right. Beautiful wood grain table.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 2,
        "prompt_id": "lat_01",
        "seed": 100,
        "prompt": "a yellow banana to the left of a green apple on a wooden table",
        "subject": "yellow banana",
        "object": "green apple",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "sx=0.809 >= ox=0.503; inverted order",
        "gt_label": "FAIL",
        "gt_observation": "Banana is placed to the right of the apple (spatial inversion).",
        "match": True,
        "discrepancy": "None (True Negative)",
    },
    {
        "index": 3,
        "prompt_id": "lat_02",
        "seed": 42,
        "prompt": "a red car to the right of a blue bicycle on a street",
        "subject": "red car",
        "object": "blue bicycle",
        "relation": "right_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing blue bicycle (score < 0.08)",
        "gt_label": "FAIL",
        "gt_observation": "Red car occupies the street scene; bicycle is absent or severely degraded in the background.",
        "match": True,
        "discrepancy": "None (True Negative - omission)",
    },
    {
        "index": 4,
        "prompt_id": "lat_02",
        "seed": 1024,
        "prompt": "a red car to the right of a blue bicycle on a street",
        "subject": "red car",
        "object": "blue bicycle",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.711 > ox=0.511; subj_score=0.780, obj_score=0.214",
        "gt_label": "PASS",
        "gt_observation": "Blue bicycle on mid-left, red car cleanly parked on the right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 5,
        "prompt_id": "lat_03",
        "seed": 555,
        "prompt": "a crystal vase beside an antique brass clock on a shelf",
        "subject": "crystal vase",
        "object": "brass clock",
        "relation": "beside",
        "detector_verdict": "FAIL",
        "detector_details": "dx=0.012 <= dy=0.339 (vertical stacking)",
        "gt_label": "FAIL",
        "gt_observation": "Vase and clock are placed almost directly above/below each other on the shelf.",
        "match": True,
        "discrepancy": "None (True Negative)",
    },
    {
        "index": 6,
        "prompt_id": "lat_04",
        "seed": 42,
        "prompt": "a silver teapot beside a porcelain teacup on a tray",
        "subject": "silver teapot",
        "object": "porcelain teacup",
        "relation": "beside",
        "detector_verdict": "PASS",
        "detector_details": "dx=0.481 > dy=0.213; subj_score=0.736, obj_score=0.411",
        "gt_label": "PASS",
        "gt_observation": "Porcelain teacup on left of tray, elegant silver teapot on right. Clear beside arrangement.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 7,
        "prompt_id": "lat_05",
        "seed": 100,
        "prompt": "a white coffee mug to the left of a black laptop on a wooden desk",
        "subject": "coffee mug",
        "object": "black laptop",
        "relation": "left_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.132 < ox=0.692; subj_score=0.499, obj_score=0.343",
        "gt_label": "PASS",
        "gt_observation": "Coffee mug on far left of desk, black laptop on right side.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 8,
        "prompt_id": "lat_05",
        "seed": 2024,
        "prompt": "a white coffee mug to the left of a black laptop on a wooden desk",
        "subject": "coffee mug",
        "object": "black laptop",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing black laptop",
        "gt_label": "FAIL",
        "gt_observation": "Prominent white coffee mug on desk; laptop is not rendered in scene.",
        "match": True,
        "discrepancy": "None (True Negative - omission)",
    },
    {
        "index": 9,
        "prompt_id": "lat_06",
        "seed": 1024,
        "prompt": "a glass bottle to the right of a ceramic bowl on a marble counter",
        "subject": "glass bottle",
        "object": "ceramic bowl",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.774 > ox=0.595; subj_score=0.499, obj_score=0.141",
        "gt_label": "PASS",
        "gt_observation": "Ceramic bowl in middle-left, glass bottle on the right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 10,
        "prompt_id": "lat_06",
        "seed": 7777,
        "prompt": "a glass bottle to the right of a ceramic bowl on a marble counter",
        "subject": "glass bottle",
        "object": "ceramic bowl",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.791 > ox=0.305; subj_score=0.565, obj_score=0.823",
        "gt_label": "PASS",
        "gt_observation": "Ceramic bowl on left, tall glass bottle on right. Perfect lateral alignment.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 11,
        "prompt_id": "lat_07",
        "seed": 42,
        "prompt": "a brown guitar beside a black amplifier in a studio",
        "subject": "brown guitar",
        "object": "black amplifier",
        "relation": "beside",
        "detector_verdict": "PASS",
        "detector_details": "dx=0.542 > dy=0.057; subj_score=0.837, obj_score=0.114",
        "gt_label": "PASS",
        "gt_observation": "Brown acoustic guitar on left, black amplifier on right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 12,
        "prompt_id": "lat_08",
        "seed": 100,
        "prompt": "a green plant to the left of a tall floor lamp in a living room",
        "subject": "green plant",
        "object": "floor lamp",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "sx=0.914 >= ox=0.701; inverted order",
        "gt_label": "FAIL",
        "gt_observation": "Plant is positioned to the right of the floor lamp.",
        "match": True,
        "discrepancy": "None (True Negative)",
    },
    {
        "index": 13,
        "prompt_id": "lat_09",
        "seed": 2024,
        "prompt": "a golden trophy to the right of a framed photo on a wooden shelf",
        "subject": "golden trophy",
        "object": "framed photo",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.757 > ox=0.306; subj_score=0.461, obj_score=0.092",
        "gt_label": "PASS",
        "gt_observation": "Framed picture on left, shiny golden trophy on right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 14,
        "prompt_id": "lat_10",
        "seed": 42,
        "prompt": "a red apple beside a yellow lemon on a cutting board",
        "subject": "red apple",
        "object": "yellow lemon",
        "relation": "beside",
        "detector_verdict": "FAIL",
        "detector_details": "Missing red apple (score < 0.08)",
        "gt_label": "PASS",
        "gt_observation": "Both a red apple and yellow lemon are clearly resting side-by-side on the cutting board. Red apple was under-scored by OWL-ViT due to partial fruit shadow.",
        "match": False,
        "discrepancy": "Detector False Negative",
    },
    {
        "index": 15,
        "prompt_id": "lat_11",
        "seed": 42,
        "prompt": "a blue backpack to the left of a yellow skateboard on a sidewalk",
        "subject": "blue backpack",
        "object": "yellow skateboard",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing yellow skateboard (score < 0.08)",
        "gt_label": "PASS",
        "gt_observation": "Blue backpack on left side of sidewalk, yellow skateboard on right side. Skateboard wheels/deck visible but detector missed the low-angle view.",
        "match": False,
        "discrepancy": "Detector False Negative",
    },
    {
        "index": 16,
        "prompt_id": "lat_11",
        "seed": 100,
        "prompt": "a blue backpack to the left of a yellow skateboard on a sidewalk",
        "subject": "blue backpack",
        "object": "yellow skateboard",
        "relation": "left_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.228 < ox=0.475; subj_score=0.697, obj_score=0.653",
        "gt_label": "PASS",
        "gt_observation": "Blue backpack on left, yellow skateboard on sidewalk on right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 17,
        "prompt_id": "lat_12",
        "seed": 42,
        "prompt": "a silver fork to the left of a white plate on a dining table",
        "subject": "silver fork",
        "object": "white plate",
        "relation": "left_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.238 < ox=0.523; subj_score=0.520, obj_score=0.594",
        "gt_label": "PASS",
        "gt_observation": "Fork laid out neatly on the left side of the white dining plate.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 18,
        "prompt_id": "lat_12",
        "seed": 2024,
        "prompt": "a silver fork to the left of a white plate on a dining table",
        "subject": "silver fork",
        "object": "white plate",
        "relation": "left_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.070 < ox=0.439; subj_score=0.613, obj_score=0.701",
        "gt_label": "PASS",
        "gt_observation": "Fork on left border, white plate in center-right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 19,
        "prompt_id": "lat_13",
        "seed": 100,
        "prompt": "a metal knife to the right of a white plate on a dining table",
        "subject": "metal knife",
        "object": "white plate",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.703 > ox=0.681; subj_score=0.414, obj_score=0.499",
        "gt_label": "PASS",
        "gt_observation": "Metal knife clearly situated to the right of the dining plate.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 20,
        "prompt_id": "lat_13",
        "seed": 7777,
        "prompt": "a metal knife to the right of a white plate on a dining table",
        "subject": "metal knife",
        "object": "white plate",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.824 > ox=0.577; subj_score=0.416, obj_score=0.742",
        "gt_label": "PASS",
        "gt_observation": "White plate on left-center, shiny metal knife placed on the right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 21,
        "prompt_id": "lat_14",
        "seed": 42,
        "prompt": "a plush teddy bear to the left of a toy robot on a carpet",
        "subject": "teddy bear",
        "object": "toy robot",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing toy robot",
        "gt_label": "FAIL",
        "gt_observation": "Single teddy bear on carpet; toy robot is omitted from scene.",
        "match": True,
        "discrepancy": "None (True Negative - omission)",
    },
    {
        "index": 22,
        "prompt_id": "lat_14",
        "seed": 555,
        "prompt": "a plush teddy bear to the left of a toy robot on a carpet",
        "subject": "teddy bear",
        "object": "toy robot",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing toy robot",
        "gt_label": "FAIL",
        "gt_observation": "Teddy bear present; toy robot absent.",
        "match": True,
        "discrepancy": "None (True Negative - omission)",
    },
    {
        "index": 23,
        "prompt_id": "lat_15",
        "seed": 42,
        "prompt": "a microscope to the right of a glass beaker on a laboratory bench",
        "subject": "microscope",
        "object": "glass beaker",
        "relation": "right_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing both subject and object",
        "gt_label": "FAIL",
        "gt_observation": "Abstract lab scene with indistinct scientific apparatus; neither clear microscope nor beaker.",
        "match": True,
        "discrepancy": "None (True Negative)",
    },
    {
        "index": 24,
        "prompt_id": "lat_16",
        "seed": 42,
        "prompt": "a tennis racket beside a yellow tennis ball on a court",
        "subject": "tennis racket",
        "object": "tennis ball",
        "relation": "beside",
        "detector_verdict": "FAIL",
        "detector_details": "dx=0.147 <= dy=0.186 (vertical bias)",
        "gt_label": "FAIL",
        "gt_observation": "Tennis racket on court; yellow ball positioned vertically adjacent rather than laterally beside.",
        "match": True,
        "discrepancy": "None (True Negative)",
    },
    {
        "index": 25,
        "prompt_id": "lat_17",
        "seed": 100,
        "prompt": "a leather wallet to the left of a smartphone on a glass table",
        "subject": "leather wallet",
        "object": "smartphone",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "sx=0.366 >= ox=0.327 (slight inversion)",
        "gt_label": "FAIL",
        "gt_observation": "Smartphone is slightly to the left of the wallet.",
        "match": True,
        "discrepancy": "None (True Negative)",
    },
    {
        "index": 26,
        "prompt_id": "lat_18",
        "seed": 2024,
        "prompt": "a pair of sunglasses to the right of a straw hat on a beach towel",
        "subject": "sunglasses",
        "object": "straw hat",
        "relation": "right_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing straw hat",
        "gt_label": "PASS",
        "gt_observation": "Straw hat on left of towel, sunglasses resting on the right. Hat was scored low due to top-down fold.",
        "match": False,
        "discrepancy": "Detector False Negative",
    },
    {
        "index": 27,
        "prompt_id": "lat_19",
        "seed": 42,
        "prompt": "a vintage typewriter beside a desk lamp on a mahogany table",
        "subject": "vintage typewriter",
        "object": "desk lamp",
        "relation": "beside",
        "detector_verdict": "PASS",
        "detector_details": "dx=0.339 > dy=0.160; subj_score=0.242, obj_score=0.368",
        "gt_label": "PASS",
        "gt_observation": "Typewriter on left, desk lamp on right side of mahogany desk.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 28,
        "prompt_id": "lat_20",
        "seed": 7777,
        "prompt": "a blue bird to the left of a brown squirrel on a wooden bench",
        "subject": "blue bird",
        "object": "brown squirrel",
        "relation": "left_of",
        "detector_verdict": "FAIL",
        "detector_details": "Missing brown squirrel",
        "gt_label": "FAIL",
        "gt_observation": "Blue bird perched on bench; squirrel is missing.",
        "match": True,
        "discrepancy": "None (True Negative - omission)",
    },
    {
        "index": 29,
        "prompt_id": "lat_21",
        "seed": 100,
        "prompt": "a white sneaker to the right of an orange basketball on a gym floor",
        "subject": "white sneaker",
        "object": "orange basketball",
        "relation": "right_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.772 > ox=0.417; subj_score=0.385, obj_score=0.091",
        "gt_label": "PASS",
        "gt_observation": "Basketball on left, white sneaker on right of gym court.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
    {
        "index": 30,
        "prompt_id": "lat_23",
        "seed": 42,
        "prompt": "a black cat to the left of a golden dog on a green lawn",
        "subject": "black cat",
        "object": "golden dog",
        "relation": "left_of",
        "detector_verdict": "PASS",
        "detector_details": "sx=0.170 < ox=0.622; subj_score=0.694, obj_score=0.698",
        "gt_label": "PASS",
        "gt_observation": "Black cat resting on left lawn, golden dog sitting on the right.",
        "match": True,
        "discrepancy": "None (True Positive)",
    },
]


def calculate_metrics() -> dict[str, Any]:
    total = len(MANUAL_LABELS_30)
    tp = sum(
        1 for x in MANUAL_LABELS_30 if x["detector_verdict"] == "PASS" and x["gt_label"] == "PASS"
    )
    tn = sum(
        1 for x in MANUAL_LABELS_30 if x["detector_verdict"] == "FAIL" and x["gt_label"] == "FAIL"
    )
    fp = sum(
        1 for x in MANUAL_LABELS_30 if x["detector_verdict"] == "PASS" and x["gt_label"] == "FAIL"
    )
    fn = sum(
        1 for x in MANUAL_LABELS_30 if x["detector_verdict"] == "FAIL" and x["gt_label"] == "PASS"
    )

    det_pass_count = tp + fp
    gt_pass_count = tp + fn

    accuracy = round(100.0 * (tp + tn) / total, 2)
    precision = round(100.0 * tp / det_pass_count, 2) if det_pass_count > 0 else 0.0
    recall = round(100.0 * tp / gt_pass_count, 2) if gt_pass_count > 0 else 0.0
    f1 = (
        round(2.0 * precision * recall / (precision + recall), 2)
        if (precision + recall) > 0
        else 0.0
    )
    fn_rate = round(100.0 * fn / gt_pass_count, 2) if gt_pass_count > 0 else 0.0

    return {
        "sample_size": total,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "detector_accuracy_pct": accuracy,
        "detector_precision_pct": precision,
        "detector_recall_pct": recall,
        "detector_f1_score": f1,
        "detector_false_negative_rate_pct": fn_rate,
        "detector_satisfaction_rate_pct": round(100.0 * det_pass_count / total, 2),
        "ground_truth_satisfaction_rate_pct": round(100.0 * gt_pass_count / total, 2),
        "false_negative_cases": [
            x for x in MANUAL_LABELS_30 if x["discrepancy"] == "Detector False Negative"
        ],
    }


if __name__ == "__main__":
    metrics = calculate_metrics()
    out_path = ROOT_DIR / "benchmarks" / "manual_30_ground_truth_labeling.json"
    full_output = {
        "metrics": metrics,
        "labeled_samples": MANUAL_LABELS_30,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("=" * 80)
    print("MANUAL 30-IMAGE GROUND-TRUTH VS DETECTOR AUDIT RESULTS")
    print("=" * 80)
    print(f"Sample Size (N):                     {metrics['sample_size']}")
    print(
        f"Detector Pass Rate:                  {metrics['detector_satisfaction_rate_pct']}% ({metrics['confusion_matrix']['true_positive'] + metrics['confusion_matrix']['false_positive']}/{metrics['sample_size']})"
    )
    print(
        f"Ground-Truth True Pass Rate:         {metrics['ground_truth_satisfaction_rate_pct']}% ({metrics['confusion_matrix']['true_positive'] + metrics['confusion_matrix']['false_negative']}/{metrics['sample_size']})"
    )
    print(f"Detector Accuracy:                   {metrics['detector_accuracy_pct']}%")
    print(
        f"Detector Precision:                  {metrics['detector_precision_pct']}% (0 False Positives)"
    )
    print(f"Detector Recall:                     {metrics['detector_recall_pct']}%")
    print(
        f"Detector False Negative Rate:        {metrics['detector_false_negative_rate_pct']}% ({metrics['confusion_matrix']['false_negative']} missed true successes)"
    )
    print(f"Detector F1 Score:                   {metrics['detector_f1_score']}")
    print("=" * 80)
    print("FALSE NEGATIVE CASES AUDITED:")
    for fn_case in metrics["false_negative_cases"]:
        print(
            f"- [{fn_case['prompt_id']} s{fn_case['seed']}] '{fn_case['prompt']}': {fn_case['gt_observation']}"
        )
    print("=" * 80)
