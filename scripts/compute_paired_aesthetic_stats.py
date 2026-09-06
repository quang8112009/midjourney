import json
import math
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent

with open(ROOT_DIR / "benchmarks" / "aesthetic_baseline.json") as f:
    sd15_data = json.load(f)

with open(ROOT_DIR / "benchmarks" / "sd35m_aesthetic_baseline.json") as f:
    sd35_data = json.load(f)

prompts_sd15 = (
    sd15_data["per_prompt_results"]
    if "per_prompt_results" in sd15_data
    else sd15_data.get("results_per_prompt", [])
)
prompts_sd35 = sd35_data["matched_512x512_20steps"]["per_prompt_results"]

metrics = [
    ("laion_aesthetic_v2_4", "LAION v2.4"),
    ("clip_alignment", "CLIP Alignment"),
    ("pickscore_v1", "PickScore v1"),
    ("hps_v2_1", "HPS v2.1"),
    ("imagereward", "ImageReward"),
]

paired_data = {m[0]: {"sd15": [], "sd35": []} for m in metrics}

for p15, p35 in zip(prompts_sd15, prompts_sd35, strict=True):
    assert p15["prompt_id"] == p35["prompt_id"]
    seeds_15 = p15.get("seeds") or p15.get("per_seed_runs", [])
    seeds_35 = p35.get("seeds") or p35.get("per_seed_runs", [])
    for s15, s35 in zip(seeds_15, seeds_35, strict=True):
        assert s15["seed"] == s35["seed"]
        for m_key, _ in metrics:
            paired_data[m_key]["sd15"].append(s15[m_key])
            paired_data[m_key]["sd35"].append(s35[m_key])


print("=" * 115)
print(f"{'Metric':<18} | {'SD v1.5 (Mean+-std)':<18} | {'SD 3.5 M (Mean+-std)':<18} | {'Mean Diff':<12} | {'95% CI of Diff':<18} | {'Paired t-stat':<14} | {'p-value':<12}")
print("=" * 115)

stats_summary = {}

for m_key, m_name in metrics:
    v15 = np.array(paired_data[m_key]["sd15"])
    v35 = np.array(paired_data[m_key]["sd35"])
    diff = v35 - v15
    n = len(diff)
    
    mean_15 = float(np.mean(v15))
    std_15 = float(np.mean([np.std(v15[i*4:(i+1)*4]) for i in range(40)])) # within-prompt cross-seed std
    
    mean_35 = float(np.mean(v35))
    std_35 = float(np.mean([np.std(v35[i*4:(i+1)*4]) for i in range(40)])) # within-prompt cross-seed std
    
    d_mean = float(np.mean(diff))
    d_std = float(np.std(diff, ddof=1))
    d_se = d_std / math.sqrt(n)
    
    t_stat = d_mean / d_se
    # Normal approximation p-value for N=160
    from math import erfc
    p_val = erfc(abs(t_stat) / math.sqrt(2))
    
    ci_low = d_mean - 1.96 * d_se
    ci_high = d_mean + 1.96 * d_se
    
    stats_summary[m_key] = {
        "sd15_mean": round(mean_15, 4),
        "sd15_seed_std": round(std_15, 4),
        "sd35_mean": round(mean_35, 4),
        "sd35_seed_std": round(std_35, 4),
        "d_mean": round(d_mean, 4),
        "d_se": round(d_se, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "t_stat": round(t_stat, 3),
        "p_value": p_val,
    }
    
    p_str = f"p={p_val:.2e}" if p_val < 1e-4 else f"p={p_val:.4f}"
    print(f"{m_name:<18} | {mean_15:6.4f} +- {std_15:6.4f}   | {mean_35:6.4f} +- {std_35:6.4f}   | {d_mean:+8.4f}   | [{ci_low:+7.4f}, {ci_high:+7.4f}] | t={t_stat:+7.2f}     | {p_str:<12}")

print("=" * 115)

with open(ROOT_DIR / "benchmarks" / "paired_backbone_aesthetic_stats.json", "w") as f:
    json.dump(stats_summary, f, indent=2)
