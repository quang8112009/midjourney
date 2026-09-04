"""Object Presence Metric & Detector Ground-Truth Analysis for Lateral Spatial Guidance.

Evaluates entity presence (0, 1, or 2 entities detected) across strengths 0.00, 1.50, 3.00, 6.00
on the existing 768 images from the lateral dedicated benchmark (24 prompts x 8 seeds = 192 pairs).
Generates an audit of detector false negatives and failure modes.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS, SEEDS_192  # noqa: E402
from scripts.eval_spatial_rigorous_benchmark import wilson_score_interval  # noqa: E402


class PresenceAndDetectorEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        owl_dir = ROOT_DIR / "models" / "owlvit_base_patch32"
        self.owl_processor = OwlViTProcessor.from_pretrained(str(owl_dir))
        self.owl_model = OwlViTForObjectDetection.from_pretrained(str(owl_dir)).to(device).eval()

    @torch.inference_mode()
    def detect_entities(
        self, image: Image.Image, subject_text: str, object_text: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        queries = [f"a {subject_text}", f"a {object_text}"]
        inputs = self.owl_processor(text=[queries], images=image, return_tensors="pt").to(
            self.device
        )
        outputs = self.owl_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.owl_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.08
        )[0]

        best_subj = None
        best_obj = None

        for idx, label_idx in enumerate(results["labels"].tolist()):
            score = float(results["scores"][idx].item())
            box = [float(x) for x in results["boxes"][idx].tolist()]
            w, h = image.size
            norm_box = [box[1] / h, box[0] / w, box[3] / h, box[2] / w]
            cy = (norm_box[0] + norm_box[2]) / 2.0
            cx = (norm_box[1] + norm_box[3]) / 2.0

            det = {
                "score": round(score, 3),
                "box": [round(x, 4) for x in norm_box],
                "center": (round(cy, 4), round(cx, 4)),
            }
            if label_idx == 0 and (best_subj is None or score > best_subj["score"]):
                best_subj = det
            elif label_idx == 1 and (best_obj is None or score > best_obj["score"]):
                best_obj = det

        return best_subj, best_obj

    def check_relation(
        self, subj_det: dict[str, Any] | None, obj_det: dict[str, Any] | None, relation: str
    ) -> tuple[bool, str, str]:
        """Returns (satisfied, reason, failure_category).
        failure_category is one of: 'success', 'omission_subject', 'omission_object',
        'omission_both', 'misplacement'.
        """
        if subj_det is None and obj_det is None:
            return False, "Missing: subject, object", "omission_both"
        if subj_det is None:
            return False, "Missing: subject", "omission_subject"
        if obj_det is None:
            return False, "Missing: object", "omission_object"

        # Both entities are present, check geometry
        sy, sx = subj_det["center"]
        oy, ox = obj_det["center"]

        if relation == "left_of":
            if sx < ox:
                return True, f"Subject x ({sx:.3f}) < Object x ({ox:.3f})", "success"
            return False, f"Inverted: Subject x ({sx:.3f}) >= Object x ({ox:.3f})", "misplacement"

        elif relation == "right_of":
            if sx > ox:
                return True, f"Subject x ({sx:.3f}) > Object x ({ox:.3f})", "success"
            return False, f"Inverted: Subject x ({sx:.3f}) <= Object x ({ox:.3f})", "misplacement"

        elif relation == "beside":
            dx = abs(sx - ox)
            dy = abs(sy - oy)
            if dx > dy:
                return True, f"Beside: dx ({dx:.3f}) > dy ({dy:.3f})", "success"
            return False, f"Vertical alignment: dx ({dx:.3f}) <= dy ({dy:.3f})", "misplacement"

        return False, f"Unknown relation {relation}", "unknown"


def run_presence_evaluation() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running presence evaluation on {device}...")
    evaluator = PresenceAndDetectorEvaluator(device)

    img_dir = ROOT_DIR / "benchmarks" / "images" / "lateral_dedicated_n192"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory {img_dir} does not exist.")

    strengths = [0.00, 1.50, 3.00, 6.00]
    results_by_strength: dict[float, list[dict[str, Any]]] = {s: [] for s in strengths}

    for spec in LATERAL_24_SPECS:
        pid = spec["id"]
        for seed in SEEDS_192:
            for s in strengths:
                filename = f"{pid}_s{seed}_str_{s:.2f}.png"
                img_path = img_dir / filename
                if not img_path.exists():
                    print(f"Warning: Missing {img_path}")
                    continue

                image = Image.open(img_path).convert("RGB")
                subj_det, obj_det = evaluator.detect_entities(
                    image, spec["subject"], spec["object"]
                )
                sat, reason, fail_cat = evaluator.check_relation(
                    subj_det, obj_det, spec["relation"]
                )

                entities_present = (1 if subj_det is not None else 0) + (
                    1 if obj_det is not None else 0
                )

                record = {
                    "prompt_id": pid,
                    "prompt": spec["prompt"],
                    "subject": spec["subject"],
                    "object": spec["object"],
                    "relation": spec["relation"],
                    "seed": seed,
                    "strength": s,
                    "image_path": str(img_path.relative_to(ROOT_DIR)),
                    "subj_det": subj_det,
                    "obj_det": obj_det,
                    "entities_present": entities_present,
                    "both_present": entities_present == 2,
                    "satisfied": sat,
                    "reason": reason,
                    "failure_category": fail_cat,
                }
                results_by_strength[s].append(record)

    summary: dict[str, Any] = {}
    for s, records in results_by_strength.items():
        n = len(records)
        total_prompted_entities = n * 2
        total_detected_entities = sum(r["entities_present"] for r in records)
        both_present_count = sum(1 for r in records if r["both_present"])
        sat_count = sum(1 for r in records if r["satisfied"])

        omitted_subj_count = sum(1 for r in records if r["failure_category"] == "omission_subject")
        omitted_obj_count = sum(1 for r in records if r["failure_category"] == "omission_object")
        omitted_both_count = sum(1 for r in records if r["failure_category"] == "omission_both")
        misplaced_count = sum(1 for r in records if r["failure_category"] == "misplacement")

        entity_presence_pct = round(100.0 * total_detected_entities / total_prompted_entities, 2)
        dual_presence_pct = round(100.0 * both_present_count / n, 2)
        satisfaction_pct = round(100.0 * sat_count / n, 2)

        entity_presence_ci = wilson_score_interval(total_detected_entities, total_prompted_entities)
        dual_presence_ci = wilson_score_interval(both_present_count, n)
        sat_ci = wilson_score_interval(sat_count, n)

        summary[f"strength_{s:.2f}"] = {
            "strength": s,
            "n_runs": n,
            "entity_presence_count": f"{total_detected_entities}/{total_prompted_entities}",
            "entity_presence_pct": entity_presence_pct,
            "entity_presence_ci_95": entity_presence_ci,
            "dual_presence_count": f"{both_present_count}/{n}",
            "dual_presence_pct": dual_presence_pct,
            "dual_presence_ci_95": dual_presence_ci,
            "satisfaction_count": f"{sat_count}/{n}",
            "satisfaction_pct": satisfaction_pct,
            "satisfaction_ci_95": sat_ci,
            "breakdown": {
                "success": sat_count,
                "misplaced": misplaced_count,
                "omission_subject_only": omitted_subj_count,
                "omission_object_only": omitted_obj_count,
                "omission_both": omitted_both_count,
                "total_omissions": omitted_subj_count + omitted_obj_count + omitted_both_count,
            },
        }

    return {
        "metadata": {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "device": str(device),
            "n_prompts": len(LATERAL_24_SPECS),
            "n_seeds": len(SEEDS_192),
            "total_images": sum(len(r) for r in results_by_strength.values()),
        },
        "summary": summary,
        "detailed_records": {f"strength_{s:.2f}": results_by_strength[s] for s in strengths},
    }


if __name__ == "__main__":
    data = run_presence_evaluation()
    out_file = ROOT_DIR / "benchmarks" / "lateral_presence_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Results written to {out_file}")

    print("\n" + "=" * 80)
    print("OBJECT-PRESENCE & SATISFACTION SUMMARY (LATERAL N=192)")
    print("=" * 80)
    hdr = f"{'Strength':<10} | {'Entity Presence':<18} | {'Both Present':<18} | {'Satisfaction':<16} | {'Misplaced':<10} | {'Omitted':<10}"  # noqa: E501
    print(hdr)
    print("-" * 80)
    for _s_key, s_data in data["summary"].items():
        st = f"{s_data['strength']:.2f}"
        ep = f"{s_data['entity_presence_pct']}% ({s_data['entity_presence_count']})"
        dp = f"{s_data['dual_presence_pct']}% ({s_data['dual_presence_count']})"
        sat = f"{s_data['satisfaction_pct']}% ({s_data['satisfaction_count']})"
        misp = s_data["breakdown"]["misplaced"]
        omit = s_data["breakdown"]["total_omissions"]
        print(f"{st:<10} | {ep:<18} | {dp:<18} | {sat:<16} | {misp:<10} | {omit:<10}")
    print("=" * 80)
