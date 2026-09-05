from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

TEST_SAME_CLASS_PROMPTS = [
    {
        "id": "sc_01",
        "prompt": "a blue ceramic mug to the left of a red ceramic mug on a wooden table",
        "subject": "blue ceramic mug",
        "object": "red ceramic mug",
        "relation": "left_of",
    },
    {
        "id": "sc_02",
        "prompt": "a red apple to the left of a green apple on a white plate",
        "subject": "red apple",
        "object": "green apple",
        "relation": "left_of",
    },
    {
        "id": "sc_03",
        "prompt": "a yellow car to the right of a black car on an asphalt road",
        "subject": "yellow car",
        "object": "black car",
        "relation": "right_of",
    },
    {
        "id": "sc_04",
        "prompt": "a gold candle to the left of a silver candle on a mantelpiece",
        "subject": "gold candle",
        "object": "silver candle",
        "relation": "left_of",
    },
    {
        "id": "sc_05",
        "prompt": "a brown leather boot to the right of a black leather boot on a floor",
        "subject": "brown leather boot",
        "object": "black leather boot",
        "relation": "right_of",
    },
    {
        "id": "sc_06",
        "prompt": "a purple book to the left of an orange book on a wooden desk",
        "subject": "purple book",
        "object": "orange book",
        "relation": "left_of",
    },
]

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    # box format: [ymin, xmin, ymax, xmax]
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0.0, yB - yA) * max(0.0, xB - xA)
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def main():
    print("=" * 80)
    print("VALIDATING OWL-ViT DETECTOR ON SAME-CLASS ATTRIBUTE-DIFFERENTIATED PAIRS")
    print("=" * 80)

    out_dir = ROOT_DIR / "benchmarks" / "images" / "same_class_detector_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading SD 3.5 Medium to generate test pairs...")
    pipe = StableDiffusion3Pipeline.from_pretrained("D:/midjourney/models/sd35_medium", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    print("\n2. Loading OWL-ViT Object Detector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owl_processor = OwlViTProcessor.from_pretrained("models/owlvit_base_patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("models/owlvit_base_patch32").to(device).eval()

    generated_cases = []
    print("\n3. Generating and Evaluating 6 Same-Class Pairs (Seeds 42 & 100 = 12 total images)...")

    for spec in TEST_SAME_CLASS_PROMPTS:
        pid = spec["id"]
        prompt = spec["prompt"]
        subj = spec["subject"]
        obj = spec["object"]

        for seed in [42, 100]:
            img_path = out_dir / f"{pid}_s{seed}.png"
            if not img_path.exists():
                gen = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    img = pipe(
                        prompt=prompt,
                        num_inference_steps=20,
                        guidance_scale=4.5,
                        width=512,
                        height=512,
                        generator=gen,
                    ).images[0]
                img.save(img_path)
            else:
                img = Image.open(img_path).convert("RGB")

            # Run OWL-ViT with color-qualified queries
            queries = [f"a {subj}", f"a {obj}"]
            inputs = owl_processor(text=[queries], images=img, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = owl_model(**inputs)
            target_sizes = torch.tensor([img.size[::-1]]).to(device)
            results = owl_processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.08
            )[0]

            best_subj = None
            best_obj = None

            for idx, label_idx in enumerate(results["labels"].tolist()):
                score = float(results["scores"][idx].item())
                box = [float(x) for x in results["boxes"][idx].tolist()]
                w, h = img.size
                norm_box = [box[1] / h, box[0] / w, box[3] / h, box[2] / w]
                det = {"score": round(score, 3), "box": [round(x, 4) for x in norm_box], "center": ((norm_box[0]+norm_box[2])/2, (norm_box[1]+norm_box[3])/2)}
                if label_idx == 0 and (best_subj is None or score > best_subj["score"]):
                    best_subj = det
                elif label_idx == 1 and (best_obj is None or score > best_obj["score"]):
                    best_obj = det

            both_detected = best_subj is not None and best_obj is not None
            iou = compute_iou(best_subj["box"], best_obj["box"]) if both_detected else 0.0
            distinct_boxes = both_detected and iou < 0.60

            generated_cases.append({
                "id": pid,
                "prompt": prompt,
                "seed": seed,
                "subj": subj,
                "obj": obj,
                "subj_detected": best_subj is not None,
                "obj_detected": best_obj is not None,
                "subj_score": best_subj["score"] if best_subj else 0.0,
                "obj_score": best_obj["score"] if best_obj else 0.0,
                "iou": round(iou, 3),
                "distinct_boxes": distinct_boxes,
                "both_detected": both_detected,
                "subj_center": best_subj["center"] if best_subj else None,
                "obj_center": best_obj["center"] if best_obj else None,
            })

            status = "SEPARATED" if distinct_boxes else ("DUPLICATE_BOX" if iou >= 0.60 else "MISSING")
            print(f"  [{pid} s={seed}] {subj} vs {obj} -> {status} (IoU={iou:.2f}, S_score={best_subj['score'] if best_subj else 0.0:.2f}, O_score={best_obj['score'] if best_obj else 0.0:.2f})")

    # Summary Statistics
    total_imgs = len(generated_cases)
    both_det_count = sum(1 for c in generated_cases if c["both_detected"])
    distinct_count = sum(1 for c in generated_cases if c["distinct_boxes"])
    duplicate_count = sum(1 for c in generated_cases if c["both_detected"] and c["iou"] >= 0.60)
    missing_count = sum(1 for c in generated_cases if not c["both_detected"])

    print("\n" + "=" * 80)
    print("DETECTOR VALIDATION SUMMARY ON SAME-CLASS PAIRS:")
    print("=" * 80)
    print(f"Total Test Images:                   {total_imgs}")
    print(f"Dual Entity Detection Rate:          {both_det_count}/{total_imgs} ({both_det_count/total_imgs*100:.1f}%)")
    print(f"Distinct Box Separation (IoU < 0.6): {distinct_count}/{total_imgs} ({distinct_count/total_imgs*100:.1f}%)")
    print(f"Duplicate Box Failure (IoU >= 0.6):  {duplicate_count}/{total_imgs} ({duplicate_count/total_imgs*100:.1f}%)")
    print(f"Missing Entity Detection:            {missing_count}/{total_imgs} ({missing_count/total_imgs*100:.1f}%)")

if __name__ == "__main__":
    main()
