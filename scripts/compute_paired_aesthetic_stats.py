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
    ("imagereward", "ImageReward"),
    ("hps_v2_1", "HPS v2.1"),
    ("clip_alignment", "CLIP Alignment"),
    ("pickscore_v1", "PickScore v1"),
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


def wilcoxon_signed_rank(diff: np.ndarray) -> tuple[float, float, float]:
    d = diff[diff != 0]
    n = len(d)
    ranks = np.argsort(np.abs(d)) + 1
    w_pos = np.sum(ranks[d > 0])
    w_neg = np.sum(ranks[d < 0])
    w = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mean_w) / std_w
    p_val = math.erfc(abs(z) / math.sqrt(2))
    return float(w), float(z), float(p_val)


print("=" * 135)
print(
    f"{'Metric':<16} | {'SD v1.5 (Mean+-std)':<20} | {'SD 3.5 M (Mean+-std)':<20} | "
    f"{'Paired Diff (d_bar)':<20} | {'95% CI of Diff':<18} | {'Paired t-test':<18} | {'Wilcoxon Test':<18}"
)
print("=" * 135)

stats_summary = {}

for m_key, m_name in metrics:
    v15 = np.array(paired_data[m_key]["sd15"])
    v35 = np.array(paired_data[m_key]["sd35"])
    diff = v35 - v15
    n = len(diff)

    mean_15 = float(np.mean(v15))
    std_15 = float(np.mean([np.std(v15[i * 4 : (i + 1) * 4]) for i in range(40)]))

    mean_35 = float(np.mean(v35))
    std_35 = float(np.mean([np.std(v35[i * 4 : (i + 1) * 4]) for i in range(40)]))

    d_mean = float(np.mean(diff))
    d_std = float(np.std(diff, ddof=1))
    d_se = d_std / math.sqrt(n)

    t_stat = d_mean / d_se
    p_t = math.erfc(abs(t_stat) / math.sqrt(2))

    ci_low = d_mean - 1.96 * d_se
    ci_high = d_mean + 1.96 * d_se

    w_stat, w_z, p_w = wilcoxon_signed_rank(diff)
    ratio_to_noise = abs(d_mean) / std_15

    stats_summary[m_key] = {
        "sd15_mean": round(mean_15, 4),
        "sd15_seed_std": round(std_15, 4),
        "sd35_mean": round(mean_35, 4),
        "sd35_seed_std": round(std_35, 4),
        "d_mean": round(d_mean, 4),
        "d_se": round(d_se, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "ratio_to_noise": round(ratio_to_noise, 2),
        "t_stat": round(t_stat, 3),
        "t_pvalue": p_t,
        "wilcoxon_z": round(w_z, 3),
        "wilcoxon_pvalue": p_w,
    }

    p_t_str = f"p={p_t:.2e}" if p_t < 1e-4 else f"p={p_t:.4f}"
    p_w_str = f"p={p_w:.2e}" if p_w < 1e-4 else f"p={p_w:.4f}"
    print(
        f"{m_name:<16} | {mean_15:6.4f} +- {std_15:6.4f}   | {mean_35:6.4f} +- {std_35:6.4f}   | "
        f"{d_mean:+8.4f} (r={ratio_to_noise:.2f}x) | [{ci_low:+7.4f}, {ci_high:+7.4f}] | "
        f"t={t_stat:+6.2f} ({p_t_str}) | z={w_z:+6.2f} ({p_w_str})"
    )

print("=" * 135)

with open(ROOT_DIR / "benchmarks" / "paired_backbone_aesthetic_stats.json", "w") as f:
    json.dump(stats_summary, f, indent=2)
