# Benchmark Datasets Schema & Directory Specification

This directory houses all raw empirical evaluation datasets, paired statistical reports, blinded validation keys, and cryptographic hashes for all experiments reported in the technical paper.

---

## 1. Master Benchmark File Manifest

| File | Experiment / Purpose | Seeds | Sample Size | Primary Metrics |
| :--- | :--- | :---: | :---: | :--- |
| `paired_backbone_8seed_stats.json` | 8-seed multi-backbone comparison (SD 3.5 M vs SD v1.5) | `SEEDS_8` | $N=320$ pairs | LAION, ImageReward, HPSv2.1, CLIP, PickScore |
| `pixart_aesthetic_baseline.json` | PixArt-Alpha 8-seed baseline dataset | `SEEDS_8` | $N=320$ images | All 5 scorers + SHA-256 hashes |
| `pixart_style_expansion_ab_results.json` | PixArt-Alpha style expansion paired A/B results | `SEEDS_8` | $N=320$ pairs | Paired $\bar{d}$, 95% CIs, Wilcoxon $p$-values |
| `sd15_8seed_aesthetic_and_ab.json` | SD v1.5 8-seed baseline & style expansion dataset | `SEEDS_8` | $N=320$ pairs | All 5 scorers + SHA-256 hashes |
| `sd35m_8seed_aesthetic_and_ab.json` | SD 3.5 Medium 8-seed baseline & expansion dataset | `SEEDS_8` | $N=320$ pairs | All 5 scorers + SHA-256 hashes |
| `human_depth_validation_blinded.json` | Blinded 120-item depth validation labeling sheet | `SEEDS_5` | $N=120$ images | Blind IDs, prompts, questions |
| `human_depth_validation_secret_key.json` | Secret key with 2D & Depth Anything V2 metrics | `SEEDS_5` | $N=120$ images | 2D vs 3D metric verdicts |
| `human_labels.json` | Ground-truth human annotations for depth validation | `SEEDS_5` | $N=120$ images | Boolean ground-truth labels |
| `depth_human_validation_results.json` | Human vs 2D vs Depth Anything V2 accuracy report | `SEEDS_5` | $N=120$ samples | Accuracy, Precision, Recall, F1 |
| `inference_time_sweep_results.json` | Sequential 4-stage inference-time sweep on SD 3.5 | `SEEDS_4` | $N=1,120$ runs | Sampler, Steps, CFG Rescale, Refiner |
| `cfg_rescale_extended_sweep.json` | Extended CFG rescale sweep ($\phi \in [0.50, 1.00]$) | `SEEDS_4` | $N=160$ pairs | LAION, ImageReward, HPSv2.1, CLIP |
| `pixart_standard24_results.json` | PixArt-Alpha Standard 24 lateral powered study | `SEEDS_192` | $N=192$ pairs | 36.8% -> 86.8% (p = 1.87e-14) |
| `pixart_hard24_results.json` | PixArt-Alpha Hard 24 lateral powered study | `SEEDS_192` | $N=192$ pairs | 33.3% -> 71.9% (p = 4.48e-17) |
| `sd15_hard24_results.json` | SD v1.5 Hard 24 lateral powered study | `SEEDS_192` | $N=192$ pairs | 18.2% -> 45.8% (p = 3.24e-11) |
| `style_expansion_regression_report.json` | Spatial coordinate invariance verification | `SEEDS_8` | 64 spatial prompts | 100% $\Delta \mu = 0.0000$ invariant proof |


---

## 2. Seed Definitions

* **Standard 4-Seed Suite (`SEEDS_4`):** `[42, 100, 2024, 7777]`
* **Extended 8-Seed Suite (`SEEDS_8`):** `[42, 100, 2024, 7777, 123, 999, 4321, 8888]`
* **Sub-Sampled 5-Seed Suite (`SEEDS_5`):** `[42, 100, 2024, 7777, 12345]`

---

## 3. JSON Structure Specifications

### A. Paired Comparison Schema (`paired_comparison`)
```json
{
  "<metric_key>": {
    "incumbent_mean": 6.3497,
    "candidate_mean": 6.3895,
    "d_mean": 0.0398,
    "ci_95": [0.0164, 0.0633],
    "ratio_to_noise": 0.25,
    "t_stat": 3.327,
    "t_pvalue": 0.000878,
    "wilcoxon_pvalue": 0.058845
  }
}
```

### B. Per-Seed Image Record Schema (`seeds` / `records_off`)
```json
{
  "prompt_id": "aes_01",
  "seed": 42,
  "laion_aesthetic_v2_4": 6.350,
  "imagereward": 0.9793,
  "hps_v2_1": 0.3394,
  "clip_alignment": 0.3022,
  "pickscore_v1": 0.1874,
  "sha256": "74953da903869d511d9ee27bd973117ec98c177d2e736ea5a2e62688ef6f165a"
}
```

---

## 4. Image Storage & Cryptographic Verification

All generated PNG images are saved in `benchmarks/images/<experiment_name>/`.
Every image is cryptographically indexed using SHA-256 (`hashlib.sha256(open(f, "rb").read()).hexdigest()`) to ensure absolute reproducibility and prevent hash collisions.
