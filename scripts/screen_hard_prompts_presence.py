from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from scripts.hard_lateral_specs import LATERAL_HARD_24_SPECS


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
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
    print("SCREENING DUAL-ENTITY PRESENCE ON ALL 24 HARD LATERAL PROMPTS (OFF / UNGUIDED)")
    print("=" * 80)

    out_dir = ROOT_DIR / "benchmarks" / "images" / "hard_prompts_presence_screen"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = ROOT_DIR / "benchmarks" / "sd35m_prompt_embeddings.pt"
    prompt_bank = torch.load(emb_path, map_location="cpu")

    print("\n1. Loading SD 3.5 Medium Transformer & VAE on CUDA...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "models/sd35_medium",
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Generate missing images
    for spec in LATERAL_HARD_24_SPECS:
        pid = spec["id"]
        prompt = spec["prompt"]
        img_path = out_dir / f"{pid}_s42.png"
        if not img_path.exists():
            p_data = prompt_bank[prompt]
            gen = torch.Generator(device="cpu").manual_seed(42)
            with torch.inference_mode():
                img = pipe(
                    prompt_embeds=p_data["prompt_embeds"].to("cuda"),
                    pooled_prompt_embeds=p_data["pooled_prompt_embeds"].to("cuda"),
                    negative_prompt_embeds=p_data["negative_prompt_embeds"].to("cuda"),
                    negative_pooled_prompt_embeds=p_data["negative_pooled_prompt_embeds"].to("cuda"),
                    num_inference_steps=20,
                    guidance_scale=4.5,
                    width=512,
                    height=512,
                    generator=gen,
                ).images[0]
            img.save(img_path)
            print(f"  Generated {pid} in 3.5s")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("\n2. Evaluating All 24 Hard Prompts with OWL-ViT Object Detector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owl_processor = OwlViTProcessor.from_pretrained("models/owlvit_base_patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("models/owlvit_base_patch32").to(device).eval()

    results = []

    for spec in LATERAL_HARD_24_SPECS:
        pid = spec["id"]
        prompt = spec["prompt"]
        subj = spec["subject"]
        obj = spec["object"]
        rel = spec["relation"]

        img_path = out_dir / f"{pid}_s42.png"
        img = Image.open(img_path).convert("RGB")

        queries = [f"a {subj}", f"a {obj}"]
        inputs = owl_processor(text=[queries], images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = owl_model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        res = owl_processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.08
        )[0]

        best_subj = None
        best_obj = None

        for idx, label_idx in enumerate(res["labels"].tolist()):
            score = float(res["scores"][idx].item())
            box = [float(x) for x in res["boxes"][idx].tolist()]
            w, h = img.size
            norm_box = [box[1] / h, box[0] / w, box[3] / h, box[2] / w]
            det = {"score": round(score, 3), "box": [round(x, 4) for x in norm_box], "center": ((norm_box[0]+norm_box[2])/2, (norm_box[1]+norm_box[3])/2)}
            if label_idx == 0 and (best_subj is None or score > best_subj["score"]):
                best_subj = det
            elif label_idx == 1 and (best_obj is None or score > best_obj["score"]):
                best_obj = det

        both_detected = best_subj is not None and best_obj is not None
        iou = compute_iou(best_subj["box"], best_obj["box"]) if both_detected else 0.0
        distinct = both_detected and iou < 0.60

        is_sat = False
        if distinct:
            sy, sx = best_subj["center"]
            oy, ox = best_obj["center"]
            if rel == "left_of":
                is_sat = sx < ox - 0.03
            elif rel == "right_of":
                is_sat = sx > ox + 0.03

        results.append({
            "id": pid,
            "prompt": prompt,
            "relation": rel,
            "subject": subj,
            "object": obj,
            "both_present": distinct,
            "satisfied": is_sat,
            "iou": round(iou, 3),
            "subj_score": best_subj["score"] if best_subj else 0.0,
            "obj_score": best_obj["score"] if best_obj else 0.0,
        })

        pres_mark = "[PRESENT]" if distinct else "[OMISSION]"
        sat_mark = "[SAT]" if is_sat else "[---]"
        print(f"  [{pid}] {rel:<8} | {pres_mark} {sat_mark} | '{subj}' ({best_subj['score'] if best_subj else 0.0:.2f}) vs '{obj}' ({best_obj['score'] if best_obj else 0.0:.2f}) | IoU={iou:.2f}")

    total_n = len(results)
    pres_count = sum(1 for r in results if r["both_present"])
    sat_count = sum(1 for r in results if r["satisfied"])

    # Left vs Right
    left_pres = sum(1 for r in results[:12] if r["both_present"])
    left_sat = sum(1 for r in results[:12] if r["satisfied"])

    right_pres = sum(1 for r in results[12:] if r["both_present"])
    right_sat = sum(1 for r in results[12:] if r["satisfied"])

    print("\n" + "=" * 80)
    print("HARD PROMPTS SCREENING SUMMARY:")
    print("=" * 80)
    print(f"Total Hard Prompts Tested:     {total_n}")
    print(f"Overall Dual-Entity Presence:  {pres_count}/{total_n} ({pres_count/total_n*100:.1f}%)")
    print(f"Overall Spatial Satisfaction:  {sat_count}/{total_n} ({sat_count/total_n*100:.1f}%)")
    print(f"  - Left_Of Prompts (N=12):    Presence: {left_pres}/12 ({left_pres/12*100:.1f}%) | Satisfaction: {left_sat}/12 ({left_sat/12*100:.1f}%)")
    print(f"  - Right_Of Prompts (N=12):   Presence: {right_pres}/12 ({right_pres/12*100:.1f}%) | Satisfaction: {right_sat}/12 ({right_sat/12*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
