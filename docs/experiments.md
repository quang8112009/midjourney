# Empirical Experimental Record: Spatial Layout Guidance Benchmark

This document maintains the complete empirical record, statistical tests, and evidence chain for cross-attention spatial layout guidance in Stable Diffusion v1.5 on live hardware (NVIDIA RTX 4060 Ti, CUDA fp16, `torch==2.6.0+cu124`).

---

## 1. Executive Summary of Empirical Findings

| Capability / Relation Category | Baseline (OFF) | Optimal Tested Strength | Statistical Verdict (Paired McNemar) | Architectural Action | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lateral (`left_of`, `right_of`, `beside`)** | $34.90\%$ ($67/192$) | **$49.48\%$** ($95/192$) @ str 6.0 | **$p = 0.000394$** (Net $+28$ pairs) | `LATERAL_GUIDANCE_STRENGTH = 6.0` | **Validated** (6.0 provisional on visual review) |
| **Depth (`in_front_of`, `behind`) [True 3D]** | $41.67\%$ ($80/192$) | $47.92\%$ ($92/192$) @ str 6.0 | **$p = 0.080690$** (Net $+12$ pairs) | `DEPTH_RELATION_GUIDANCE_STRENGTH = 0.0` | **Negative / Unvalidated** (Not significant) |
| **Depth (`in_front_of`, `behind`) [2D Proxy]** | $50.00\%$ ($96/192$) | $60.42\%$ ($116/192$) @ str 6.0 | $p = 0.002887$ (Net $+20$ pairs) | — | *Artifact of 2D vertical framing shift* |
| **Vertical-On (`on`, `on_top_of`, `resting_on`)** | **$70.83\%$** ($17/24$) | $58.33\%$ ($14/24$) @ str 6.0 | $p = 0.453100$ (Net $-3$ pairs) | `VERTICAL_ON_GUIDANCE_STRENGTH = 0.0` | **Disabled** (Base prior is stronger) |
| **Vertical-Under (`under`, `below`)** | $45.83\%$ ($11/24$) | $58.33\%$ ($14/24$) @ str 6.0 | $p = 0.507800$ (Net $+3$ pairs) | `VERTICAL_UNDER_GUIDANCE_STRENGTH = 0.3` | **Preserved Default** (Inconclusive) |

---

## 2. Dedicated Lateral Spatial Study ($N=192$ Paired Runs, 768 Images)

To test the hypothesis that 2D attention bias acts along horizontal coordinates, a dedicated study evaluated **24 lateral prompts** $\times$ **8 seeds** (`[42, 100, 555, 1024, 2024, 7777, 9999, 12345]`) across 4 conditions: OFF (0.00), 1.50, 3.00, and 6.00.

### Paired Contingency Table vs. OFF Baseline ($67/192$, $34.90\%$)

| Condition | Satisfaction ($N=192$) | Wilson 95% CI | Both Pass ($a$) | Gain ($b$) | Loss ($c$) | Both Fail ($d$) | Net Gain ($b-c$) | McNemar Exact $p$-value | Edwards Corrected $\chi^2$ | Significant ($\alpha=0.05$)? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OFF (0.00)** | **$34.90\%$** ($67/192$) | $[28.5\%, 41.9\%]$ | — | — | — | — | — | *Baseline Reference* | — | — |
| **1.50** | **$38.02\%$** ($73/192$) | $[31.4\%, 45.1\%]$ | $61$ | $12$ | $6$ | $113$ | $+6$ | $p = 0.2379$ | $\chi^2 = 1.3889$ ($p=0.2386$) | **NO** |
| **3.00** | **$40.10\%$** ($77/192$) | $[33.4\%, 47.2\%]$ | $50$ | $27$ | $17$ | $98$ | $+10$ | $p = 0.1742$ | $\chi^2 = 1.8409$ ($p=0.1748$) | **NO** |
| **6.00** | **$49.48\%$** ($95/192$) | $[42.5\%, 56.5\%]$ | $51$ | $44$ | $16$ | $81$ | **$+28$** | **$p = 0.000394$** | $\mathbf{\chi^2 = 12.1500}$ ($p=0.000491$) | **YES ($p < 0.001$)** |

### Perceptual & Aesthetic Metrics
* **Strength 0.00:** $\text{SSIM} = 1.0000$, $\text{LAION} = 5.370$, $\text{CLIP} = 0.2741$
* **Strength 1.50:** $\text{SSIM} = 0.9055$, $\text{LAION} = 5.377$, $\text{CLIP} = 0.2735$
* **Strength 3.00:** $\text{SSIM} = 0.8368$, $\text{LAION} = 5.385$, $\text{CLIP} = 0.2743$
* **Strength 6.00:** $\text{SSIM} = 0.7215$, $\text{LAION} = 5.370$, $\text{CLIP} = 0.2761$

**Key Takeaway:** Lateral spatial steering is the one proven capability of 2D cross-attention guidance ($p = 0.000394$). Strength 3.00 does not reach statistical significance ($p = 0.1742$).

---

## 3. Dedicated Depth Study ($N=192$ Paired Runs, 576 Images)

To resolve whether depth guidance has a real physical effect, **24 depth prompts** (12 `in_front_of`, 12 `behind`) $\times$ **8 seeds** were evaluated under two distinct metrics:
1. **2D Ground-Plane Heuristic:** Evaluates relative vertical $y$-coordinates and bounding box baselines.
2. **True 3D Monocular Depth Estimator (Depth Anything V2):** Evaluates camera-space relative disparity $\bar{D}_{\text{subj}}$ vs $\bar{D}_{\text{obj}}$.

### Paired Results: 2D Ground-Plane Metric ($N=192$ Pairs)

| Condition | Satisfaction ($N=192$) | Wilson 95% CI | Both Pass ($a$) | Gain ($b$) | Loss ($c$) | Both Fail ($d$) | Net Gain ($b-c$) | McNemar Exact $p$-value | Significant? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OFF (0.00)** | **$50.00\%$** ($96/192$) | $[43.0\%, 57.0\%]$ | — | — | — | — | — | *Baseline Reference* |
| **3.00** | **$55.21\%$** ($106/192$) | $[48.1\%, 62.1\%]$ | $88$ | $18$ | $8$ | $78$ | $+10$ | $p = 0.0755$ (Not Signif.) |
| **6.00** | **$60.42\%$** ($116/192$) | $[53.4\%, 67.1\%]$ | $85$ | $31$ | $11$ | $65$ | **$+20$** | **$p = 0.002887$ (Signif.)** |

### Paired Results: True 3D Depth Anything V2 ($N=192$ Pairs)

| Condition | Satisfaction ($N=192$) | Wilson 95% CI | Both Pass ($a$) | Gain ($b$) | Loss ($c$) | Both Fail ($d$) | Net Gain ($b-c$) | McNemar Exact $p$-value | Significant? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OFF (0.00)** | **$41.67\%$** ($80/192$) | $[34.9\%, 48.7\%]$ | — | — | — | — | — | *Baseline Reference* |
| **3.00** | **$45.31\%$** ($87/192$) | $[38.4\%, 52.4\%]$ | $69$ | $18$ | $11$ | $94$ | $+7$ | $p = 0.2649$ (Not Signif.) |
| **6.00** | **$47.92\%$** ($92/192$) | $[41.0\%, 55.0\%]$ | $66$ | $26$ | $14$ | $86$ | **$+12$** | **$p = 0.080690$ (Not Signif.)** |

**Key Takeaway:** The apparent gain on the 2D depth metric ($p = 0.0029$) is an artifact of 2D vertical footing displacement (a pictorial depth cue). When measured with a true 3D monocular depth estimator, depth guidance does not achieve statistical significance ($p = 0.081$).

---

## 4. Re-scoring Existing Depth Dataset ($N=24$ per Condition, 144 Images)

Re-scoring the original 4-prompt $\times$ 6-seed benchmark images across all 6 strength sweeps using Depth Anything V2:

| Strength | 2D Ground-Plane Metric ($N=24$) | True 3D Depth Anything V2 ($N=24$) | 3D Discordant $(b, c)$ vs OFF | 3D McNemar $p$-value |
| :---: | :---: | :---: | :---: | :---: |
| **0.00 (OFF)** | $18/24$ ($75.0\%$) | $12/24$ ($50.0\%$) | *Baseline Reference* | — |
| **0.35** | $18/24$ ($75.0\%$) | $11/24$ ($45.8\%$) | $b=0, c=1$ (Net $-1$) | $p = 1.0000$ |
| **0.70** | $18/24$ ($75.0\%$) | $11/24$ ($45.8\%$) | $b=0, c=1$ (Net $-1$) | $p = 1.0000$ |
| **1.50** | $18/24$ ($75.0\%$) | $11/24$ ($45.8\%$) | $b=0, c=1$ (Net $-1$) | $p = 1.0000$ |
| **3.00** | $20/24$ ($83.3\%$) | $12/24$ ($50.0\%$) | $b=1, c=1$ (Net $+0$) | $p = 1.0000$ |
| **6.00** | $20/24$ ($83.3\%$) | $15/24$ ($62.5\%$) | $b=3, c=0$ (Net $+3$) | $p = 0.2500$ |

---

## 5. Full Mixed Benchmark Matrix (4 Categories $\times$ 6 Conditions, $N=24$ per Cell)

Evaluated on the uniform 16-prompt benchmark across 6 seeds ($N=96$ per condition, 576 total images):

```
Category Breakout Matrix (N=24 Pairs per Category per Condition)
════════════════════════════════════════════════════════════════════════════════════════════════════════════════
Category           OFF (0.00)     0.35           0.70           1.50           3.00           6.00
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Depth              18/24 (75.0%)  18/24 (75.0%)  18/24 (75.0%)  18/24 (75.0%)  20/24 (83.3%)  20/24 (83.3%)
  Discordant (b,c) —              b=0, c=0       b=0, c=0       b=0, c=0       b=2, c=0       b=2, c=0
  McNemar exact p  —              p = 1.0000     p = 1.0000     p = 1.0000     p = 0.5000     p = 0.5000

Vertical-On        17/24 (70.8%)  14/24 (58.3%)  15/24 (62.5%)  15/24 (62.5%)  16/24 (66.7%)  14/24 (58.3%)
  Discordant (b,c) —              b=0, c=3       b=0, c=2       b=1, c=3       b=2, c=3       b=2, c=5
  McNemar exact p  —              p = 0.2500     p = 0.5000     p = 0.6250     p = 1.0000     p = 0.4531

Vertical-Under     11/24 (45.8%)  10/24 (41.7%)  11/24 (45.8%)  11/24 (45.8%)  15/24 (62.5%)  14/24 (58.3%)
  Discordant (b,c) —              b=0, c=1       b=1, c=1       b=3, c=3       b=5, c=1       b=6, c=3
  McNemar exact p  —              p = 1.0000     p = 1.0000     p = 1.0000     p = 0.2188     p = 0.5078

Lateral            7/24 (29.2%)   7/24 (29.2%)   8/24 (33.3%)   9/24 (37.5%)   10/24 (41.7%)  14/24 (58.3%)
  Discordant (b,c) —              b=1, c=1       b=1, c=0       b=3, c=1       b=4, c=1       b=9, c=2
  McNemar exact p  —              p = 1.0000     p = 1.0000     p = 0.6250     p = 0.3750     p = 0.0654*
════════════════════════════════════════════════════════════════════════════════════════════════════════════════
* Note: Expanding the lateral sample from N=24 to N=192 confirmed significance: p = 0.000394.
```

---

## 6. Disclosure of Detector Audit & Depth Heuristics

### 6.1 Audit Methodology Clarification
The previously reported 30/30 detector audit was an automated algorithmic sanity check that re-evaluated OWL-ViT bounding box coordinates under relaxed geometric bounds ($s_y \ge o_y - 0.08$ vs $s_y \ge o_y - 0.05$). It did **not** constitute independent human visual labeling.

### 6.2 2D Depth Heuristic and Its Known Failure Modes
The 2D depth predicate checks vertical position in frame ($s_y \ge o_y - 0.05$ or $s_{y,\max} \ge o_{y,\max}$ for `in_front_of`) based on ground-plane perspective. Its failure modes include:
1. **Camera Angle Inversions:** Aerial shots, bird's-eye views, and upward angles invalidate ground-plane assumptions.
2. **Scale vs. Distance Ambiguity:** A large object far away vs. a small object up close cannot be resolved.
3. **Pure Occlusion without Vertical Offset:** Frontal overlapping objects at identical base coordinates cannot be judged.
4. **Suspended & Non-Planar Scenes:** Underwater, aerial, or wall-mounted objects lack ground-plane reference.

---

## 7. Visual Review & Operating Point Status (Strength 6.00)

A 20-pair contact sheet was rendered to `benchmarks/visual_review_lateral_str6.png`.

* **Mean SSIM:** $0.7373$ ($\sim 26.3\%$ structural shift).
* **Observations:** 
  * Simple two-entity scenes (`banana/apple`, `teapot/teacup`, `guitar/amplifier`) show clean lateral relocation without object duplication.
  * Complex scenes (`backpack/skateboard`, `coffee mug/laptop`) exhibit significant scene re-centering (SSIM down to $0.485$).
  * 10 of 20 sampled pairs were `Both Fail` (neither OFF nor 6.00 satisfied the prompt).
* **Operating Point Recommendation:** `LATERAL_GUIDANCE_STRENGTH = 6.0` is maintained as the default based on paired significance ($p = 0.000394$), but is flagged as **provisional** pending formal human perceptual study.
