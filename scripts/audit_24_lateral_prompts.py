from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.editing.prompt_intent import analyze_prompt
from app.services.editing.semantic_planner import plan_semantic_layout
from scripts.eval_spatial_lateral_dedicated import LATERAL_24_SPECS


def main():
    tok_clip1 = AutoTokenizer.from_pretrained("D:/midjourney/models/sd35_medium/tokenizer")
    tok_clip2 = AutoTokenizer.from_pretrained("D:/midjourney/models/sd35_medium/tokenizer_2")
    tok_t5 = AutoTokenizer.from_pretrained("D:/midjourney/models/sd35_medium/tokenizer_3")

    tokenizers = [("CLIP-L", tok_clip1), ("CLIP-G", tok_clip2), ("T5-XXL", tok_t5)]

    print(f"Auditing Token Mapping on {len(LATERAL_24_SPECS)} Lateral Benchmark Prompts...")

    total_entities_by_tok = {name: 0 for name, _ in tokenizers}
    empty_entities_by_tok = {name: 0 for name, _ in tokenizers}
    empty_details = {name: [] for name, _ in tokenizers}

    for spec in LATERAL_24_SPECS:
        pid = spec["id"]
        prompt = spec["prompt"]
        intent = analyze_prompt(prompt, mode="generate")

        for name, tok in tokenizers:
            plan = plan_semantic_layout(intent, tokenizer=tok)
            for obj in plan.objects:
                total_entities_by_tok[name] += 1
                if not obj.token_indices:
                    empty_entities_by_tok[name] += 1
                    empty_details[name].append((pid, prompt, obj.label))

    print("\n" + "=" * 80)
    print("AUDIT RESULTS (BEFORE FIX):")
    print("=" * 80)
    for name, _ in tokenizers:
        tot = total_entities_by_tok[name]
        emp = empty_entities_by_tok[name]
        print(f"[{name}] Total Entities: {tot} | Empty Tokens: {emp} ({emp/tot*100:.1f}%)")
        if emp > 0:
            print(f"  Empty Details ({emp}):")
            for pid, prompt, label in empty_details[name]:
                print(f"    - {pid}: '{label}' in \"{prompt}\"")

if __name__ == "__main__":
    main()
