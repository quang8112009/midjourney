from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval_spatial_rigorous_benchmark import wilson_score_interval


def main():
    with open("benchmarks/lateral_presence_benchmark.json") as f:
        data = json.load(f)

    for cond_key in ["strength_0.00", "strength_6.00"]:
        recs = data["detailed_records"][cond_key]

        directional_sat = 0
        directional_total = 0
        symmetric_sat = 0
        symmetric_total = 0

        left_of_sat = 0
        left_of_total = 0
        right_of_sat = 0
        right_of_total = 0

        for r in recs:
            rel = r["relation"]
            sat = r["satisfied"]
            if rel in ("left_of", "right_of"):
                directional_total += 1
                if sat:
                    directional_sat += 1
                if rel == "left_of":
                    left_of_total += 1
                    if sat:
                        left_of_sat += 1
                elif rel == "right_of":
                    right_of_total += 1
                    if sat:
                        right_of_sat += 1
            elif rel == "beside":
                symmetric_total += 1
                if sat:
                    symmetric_sat += 1

        d_ci = wilson_score_interval(directional_sat, directional_total)
        s_ci = wilson_score_interval(symmetric_sat, symmetric_total)
        all_sat = directional_sat + symmetric_sat
        all_tot = directional_total + symmetric_total
        all_ci = wilson_score_interval(all_sat, all_tot)

        print("\n=======================================================")
        print(f"SD v1.5 [Condition: {cond_key}] (N={all_tot} total):")
        print("=======================================================")
        print(f"  Overall:                   {all_sat}/{all_tot} ({all_sat/all_tot*100:.2f}%) [95% CI: {all_ci[0]:.2f}%, {all_ci[1]:.2f}%]")
        print(f"  Directional (left/right):  {directional_sat}/{directional_total} ({directional_sat/directional_total*100:.2f}%) [95% CI: {d_ci[0]:.2f}%, {d_ci[1]:.2f}%]")
        print(f"    - left_of:               {left_of_sat}/{left_of_total} ({left_of_sat/left_of_total*100:.2f}%)")
        print(f"    - right_of:              {right_of_sat}/{right_of_total} ({right_of_sat/right_of_total*100:.2f}%)")
        print(f"  Symmetric (beside):        {symmetric_sat}/{symmetric_total} ({symmetric_sat/symmetric_total*100:.2f}%) [95% CI: {s_ci[0]:.2f}%, {s_ci[1]:.2f}%]")


if __name__ == "__main__":
    main()
