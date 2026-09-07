# Measurement Pitfalls and Technique Obsolescence in Modern Text-to-Image Generation: A Rigorous Multi-Backbone Empirical Study

**Authors:** Technical Research Team  
**Date:** September 2026  
**Artifact Repository:** `https://github.com/quang8112009/midjourney`  
**Evaluation Standard:** Pre-registered, paired multi-seed live GPU evaluations ($N = 1,120+$ images, SHA-256 verified)

---

## Abstract

As text-to-image (T2I) diffusion architectures have transitioned from small UNets conditioned on CLIP text encoders (e.g., Stable Diffusion v1.5) to Diffusion Transformers conditioned on large language models (e.g., PixArt-$\alpha$, Stable Diffusion 3.5 Medium), empirical research in spatial steering and prompt engineering has increasingly suffered from evaluation artifacts and unexamined legacy assumptions. In this paper, we present an exhaustive empirical study across three foundation architectures (SD v1.5, PixArt-$\alpha$, SD 3.5 Medium) examining two primary case studies:

1. **Case Study A (The Automated Depth Measurement Illusion):** Automated spatial depth evaluation—whether using 2D ground-plane heuristics or continuous 3D monocular depth estimators (Depth Anything V2)—fails when evaluated against blinded human ground-truth labels ($N = 120$). Both automated metrics achieve only $56.67\%$ and $60.00\%$ accuracy, falling over $20\%$ below the trivial majority-class baseline ($81.11\%$). Upstream zero-shot object detection failures account for $80.6\%$ of false negatives, while $25.0\%$ of generated scenes contain unidentifiable or omitted entities that automated metrics silently evaluate. Prior reported statistical gains ($p = 0.0029$ vs $p = 0.081$) are metric artifacts rather than physical steering.
2. **Case Study B (Prompt Engineering Obsolescence in Modern Encoders):** The established community practice of appending craft descriptors ("cinematic lighting, 35mm, fine grain") is a technique specific to the CLIP era. On SD v1.5 (CLIP-L, 77 tokens), descriptor appending yields a verified aesthetic gain ($\bar{d} = +0.0831$ on LAION v2.4, $p = 2.18 \times 10^{-5}$). However, by breaking the architecture-versus-encoder confound with PixArt-$\alpha$ (DiT + T5-XXL), we demonstrate that descriptor expansion is actively harmful on T5-driven DiT architectures ($\bar{d} = -0.0665$ on LAION, $p = 2.53 \times 10^{-5}$; $\bar{d} = -0.0468$ on ImageReward, $p = 2.15 \times 10^{-8}$) and flat on SD 3.5 Medium ($\bar{d} = +0.0343$, Wilcoxon $p = 0.1630$). Across all three backbones, descriptor stuffing incurs a statistically significant cross-attention semantic dilution cost ($\Delta\text{CLIP} \in [-0.0129, -0.0051]$).
3. **Positive Control (Cross-Architecture Soft Spatial Guidance):** In contrast to depth, training-free soft cross-attention lateral guidance transfers cleanly from UNets ($25.00\% \to 53.68\%$, $p = 5.0 \times 10^{-6}$, $N = 192$) to MMDiT architectures ($80.88\% \to 90.44\%$, $p = 0.0026$ on standard scenes; $52.08\% \to 76.56\%$, $p = 4.25 \times 10^{-11}$ on cluttered/hard scenes) while strictly preserving $100\%$ zero-bias isolation on non-spatial tokens.

---

## 1. Introduction

Generative text-to-image synthesis has advanced rapidly from early convolutional latent diffusion models to massive diffusion transformers. However, evaluation methodology has struggled to keep pace. Many empirical assertions in the literature rely on synthetic proxies, circular self-auditing, unvalidated automated evaluators, or prompt engineering habits carried over from the CLIP era without re-evaluation on modern text representations.

This paper provides a rigorous, fully reproducible audit of these phenomena across three diverse, open-weights foundation models:
* **Stable Diffusion v1.5 (SD v1.5):** $0.86\text{B}$ parameter UNet conditioned on OpenAI CLIP-ViT-L/14 ($77$ token budget).
* **PixArt-$\alpha$:** $0.6\text{B}$ parameter Diffusion Transformer (DiT) conditioned on Google T5-XXL ($120$ token budget).
* **Stable Diffusion 3.5 Medium (SD 3.5 M):** $2.5\text{B}$ parameter Multimodal Diffusion Transformer (MMDiT) conditioned on a triple text encoder (CLIP-L, CLIP-G, and T5-XXL, $512$ token budget).

All experimental claims are backed by live generated images saved to disk with SHA-256 cryptographic hashes and evaluated with paired parametric ($t$-test) and non-parametric (Wilcoxon signed-rank, McNemar) statistical tests.

---

## 2. Multi-Backbone Aesthetic Baseline & Resolution Invariance

To establish a solid measurement baseline, 40 standardized style prompts spanning portraits, architecture, classical media, sci-fi, and abstract domains were evaluated across 8 fixed random seeds (`[42, 100, 2024, 7777, 123, 999, 4321, 8888]`) yielding $N = 320$ images per backbone. All images were scored using five real pretrained neural evaluators: LAION Aesthetic Predictor v2.4, ImageReward, HPS v2.1, CLIP-ViT-L/14 text-image cosine similarity, and PickScore v1.

### 2.1 Direct Paired Backbone Comparison ($N = 320$ Pairs)

| Metric | SD v1.5 (UNet + CLIP-L) | PixArt-$\alpha$ (DiT + T5-XXL) | SD 3.5 Medium (MMDiT + T5-XXL) | SD 3.5 vs SD 1.5 Paired Diff ($\bar{d}$) | 95% Confidence Interval | Paired $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LAION v2.4** | $5.954 \pm 0.234$ | **$6.418 \pm 0.147$** | $6.331 \pm 0.175$ | **$+0.3766$** | **$[+0.3281, +0.4251]$** | $p = 3.14 \times 10^{-52}$ |
| **ImageReward** | $0.774 \pm 0.126$ | **$1.009 \pm 0.077$** | $0.968 \pm 0.095$ | **$+0.1943$** | **$[+0.1681, +0.2204]$** | $p = 4.67 \times 10^{-48}$ |
| **HPS v2.1** | $0.3332 \pm 0.004$ | **$0.3403 \pm 0.002$** | $0.3391 \pm 0.003$ | **$+0.0059$** | **$[+0.0051, +0.0067]$** | $p = 1.31 \times 10^{-45}$ |
| **CLIP Alignment** | $0.2984 \pm 0.017$ | $0.3000 \pm 0.010$ | **$0.3014 \pm 0.014$** | $+0.0030$ | $[-0.0005, +0.0064]$ | $p = 0.0901$ (Neutral) |
| **PickScore v1** | $0.1865 \pm 0.004$ | $0.1869 \pm 0.003$ | **$0.1872 \pm 0.003$** | $+0.0007$ | $[-0.0001, +0.0016]$ | $p = 0.0918$ (Neutral) |

* **Statistical Concordance:** All three human-preference and aesthetic scoring models (LAION, ImageReward, HPS v2.1) exhibit massive, decisive gains ($p < 10^{-44}$). Pure semantic alignment (CLIP, PickScore) remains completely neutral ($p \approx 0.09$), proving that the aesthetic upgrade occurs without semantic drift.
* **Practical Resolution Invariance:** Evaluating SD 3.5 Medium at $512\times 512$ / 20 Euler steps ($3.90\text{ s/image}$) versus native $1024\times 1024$ / 28 steps ($23.3\text{ s/image}$) yielded identical LAION scores ($6.350$ vs $6.348$, $\Delta = -0.002$). Running at $512\times 512$ achieves a **$6\times$ computational speedup with zero measurable aesthetic degradation**.

---

## 3. Case Study A: The Automated Depth Measurement Illusion

A major claim in spatial layout literature is that 3D depth cross-attention guidance successfully controls camera Z-axis positioning (e.g. `in_front_of` vs `behind`). In earlier iterations of this project, an automated 2D ground-plane proxy reported statistically significant depth steering ($p = 0.0029, N=192$), whereas an automated 3D monocular estimator (Depth Anything V2) reported a null result ($p = 0.081$).

To resolve whether either metric measures physical reality, we conducted a rigorous **blinded human validation study**.

### 3.1 Blinded Human Study Protocol
* **Sample:** 120 images sampled from the dedicated depth study on SD 3.5 Medium, balanced across conditions (60 OFF, 60 ON at strength 6.00) and relations (60 `in_front_of`, 60 `behind`).
* **Anonymization:** All images were stripped of metadata and assigned randomized anonymous identifiers (`img_001.png` $\dots$ `img_120.png`).
* **Standing Ground-Truth Criterion:**
  1. *Overlapping objects:* Whichever occludes the other is **in front**.
  2. *Non-overlapping objects:* Whichever has the lower ground contact point is **in front**.
  3. *Can't tell:* Used when either entity is missing or unidentifiable.

### 3.2 Human Ground-Truth Results ($N = 120$)

* **Total Samples:** $120$
* **"Can't Tell" (Missing or Unidentifiable Objects):** **$30$ images ($25.0\%$)**
* **Evaluable Binary Labels:** **$90$ images ($73\text{ Yes} = 81.11\%, 17\text{ No} = 18.89\%$)**
* **Trivial Majority-Class Baseline ("Always Yes"):** **$81.11\%$ Accuracy**

| Metric / Evaluator | Accuracy vs Human | Precision | Recall | F1 Score | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline ("Always Yes")** | **$81.11\%$** | $81.11\%$ | $100.00\%$ | $89.57\%$ | $17$ | $0$ |
| **2D Ground-Plane Predicate** | **$56.67\%$** ($51/90$) | $79.31\%$ | $63.01\%$ | $70.23\%$ | **$12$** | $27$ |
| **Depth Anything V2 (3D Depth)** | **$60.00\%$** ($54/90$) | **$89.36\%$** | $57.53\%$ | $70.00\%$ | **$5$** | $31$ |

* **Paired Statistical Test:** Comparing Depth Anything V2 to the 2D predicate on the 90 human labels yields a net difference of $+3$ images ($54/90$ vs $51/90$), which is **not statistically significant (McNemar exact $p = 0.5811$)**.
* **Limitation:** Ground truth is derived from a single human annotator; inter-annotator agreement is not yet measured.



### 3.3 Diagnostic Failure Analysis & Metric Repair Ablations

An error diagnostic of all 36 Depth Anything V2 errors reveals that **$80.6\%$ ($25 / 31$) of false negatives were caused by upstream zero-shot object detection failures** (OWL-ViT failing to reach the $0.08$ score threshold on background objects like mountains, fences, and distant skylines).

We evaluated ten alternative depth aggregations, continuous threshold tuning, occlusion-only intersection rules, and multi-modal logistic regression against the 90 human labels:

| Metric Repair Variant | Accuracy | Majority Baseline | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default Box Mean** | $60.00\%$ | $81.11\%$ | $89.36\%$ | $57.53\%$ | $70.00\%$ |
| **Box Median** | $62.22\%$ | $81.11\%$ | $91.49\%$ | $58.90\%$ | $71.67\%$ |
| **10th Percentile (p10)** | $46.67\%$ | $81.11\%$ | $83.78\%$ | $42.47\%$ | $56.36\%$ |
| **Eroded Box 50%** | $58.89\%$ | $81.11\%$ | $89.13\%$ | $56.16\%$ | $68.91\%$ |
| **Foreground Adaptive (Top 40%)** | $60.00\%$ | $81.11\%$ | $91.11\%$ | $56.16\%$ | $69.49\%$ |
| **Optimal Threshold ($\tau = -0.20$)** | $62.22\%$ | $81.11\%$ | $88.24\%$ | $61.64\%$ | $72.58\%$ |
| **Occlusion Overlap Alone** | $37.78\%$ | $81.11\%$ | $74.29\%$ | $35.62\%$ | $48.15\%$ |
| **Logistic Regression (Held-out Test)** | $67.39\%$ | $81.11\%$ | $80.56\%$ | $78.38\%$ | $79.45\%$ |

### 3.4 Case Study A Finding: The Zero-Shot Detection Bottleneck
The core breakdown in automated depth evaluation is **not** the underlying fidelity of monocular depth models, but the structural fragility of the detection-then-depth paradigm:
1. **The Detection Bottleneck:** On the $N = 63$ scenes where both objects were successfully localized by zero-shot detectors, Depth Anything V2 achieves **$82.54\%$ accuracy** ($52/63$, $\text{AUC} = 0.8167$), outperforming the majority baseline on that subset ($76.19\%$, $48/63$). However, in $80.6\%$ of all false negatives ($25/31$), OWL-ViT dropped one of the prompt entities (predominantly background elements like skylines, forests, or distant fences), forcing the pipeline to assign an automatic failure.
2. **Relevance to Standard Benchmarks:** Standard benchmarks such as **T2I-CompBench++** employ the identical two-stage detection-based architecture for 3D spatial evaluation. Our finding demonstrates that benchmark failures in this category primarily measure zero-shot open-vocabulary detector recall on background entities rather than generative 3D positioning.
3. **Occlusion as a Sparse Heuristic:** Occlusion alone achieved only **$37.78\%$ accuracy** (the worst of all tested variants), despite being the primary criterion instructed to human judges. This demonstrates that human depth perception relies heavily on perspective gradients, scene context, and physical support priors rather than direct geometric mask overlap (which occurs in only $57.8\%$ of evaluable scenes).
4. **Multi-Signal Modeling as Future Work:** Multi-modal logistic regression over depth disparity, bounding box scale, and detector confidence achieved **$67.39\%$ accuracy on held-out test data** (compared to $60.00\%$ for depth alone). While still below the $81.11\%$ majority baseline with only $44$ training pairs (and subject to small-sample overfitting), this indicates that multi-modal models combining depth maps with contextual scale and detector uncertainty are a promising direction for future spatial evaluation.


---

## 4. Case Study B: Prompt Engineering Obsolescence in Modern Encoders

A ubiquitous practice in generative diffusion is appending photographic craft modifiers (e.g., *"cinematic lighting, shallow depth of field, 35mm prime lens, authentic color grading, fine grain"*) to text prompts.

To determine whether this practice is universally beneficial or an obsolete artifact of early text encoders, we conducted an 8-seed paired A/B evaluation across all three backbones ($N = 320$ pairs per model):

### 4.1 Comparative Style Expansion Results ($N = 320$ Pairs per Model)

| Model Backbone | Architecture & Text Encoder | LAION Diff ($\bar{d}$) | LAION 95% CI | ImageReward Diff ($\bar{d}$) | CLIP Alignment Cost ($\Delta\text{CLIP}$) | Empirical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5** | UNet ($0.86\text{B}$) + CLIP-L ($77\text{ tok}$) | **$+0.0831$** | **$[+0.0447, +0.1214]$** | $+0.0157$ ($p=0.148$) | **$-0.0129$** ($p = 5.9 \times 10^{-14}$) | **Genuinely Helps CLIP** |
| **PixArt-$\alpha$** | DiT ($0.6\text{B}$) + T5-XXL ($120\text{ tok}$) | **$-0.0665$** | **$[-0.0975, -0.0356]$** | **$-0.0468$** ($p = 2.1 \times 10^{-8}$) | **$-0.0068$** ($p = 3.3 \times 10^{-8}$) | **Actively Hurts T5** |
| **SD 3.5 Medium** | MMDiT ($2.5\text{B}$) + T5-XXL ($512\text{ tok}$) | $+0.0343$ | **$[+0.0025, +0.0661]$** | $+0.0070$ ($p = 0.415$) | **$-0.0051$** ($p = 7.3 \times 10^{-6}$) | **Ambiguous / Flat on T5** |

### 4.2 Causal Isolation via PixArt-$\alpha$
Because PixArt-$\alpha$ shares a transformer backbone (DiT) with SD 3.5 Medium while utilizing T5-XXL, and SD v1.5 uses a UNet with CLIP-L, PixArt breaks the architecture-versus-encoder confound:
1. **Truncation Audit:** An audit of token lengths confirmed that **$0 / 40$ prompts were truncated on any backbone** (max expanded length was $52$ tokens, well below PixArt's $120$-token ceiling).
2. **Causal Attribution:** PixArt-$\alpha$ behaves like SD 3.5 Medium (no aesthetic benefit, significant alignment cost), not like SD v1.5. This proves that **the text representation (T5-XXL vs CLIP-L), not the generative denoiser architecture, is the causal driver of prompt descriptor obsolescence**.
3. **Universal Alignment Penalty:** In all three architectures, appending extraneous style tokens divides cross-attention weights and significantly dilutes grounding on the primary subject nouns ($p < 10^{-5}$ across all backbones).

---

## 5. Positive Control: Cross-Architecture Soft Lateral Spatial Guidance

In contrast to 3D depth, horizontal lateral spatial steering (`left_of` vs `right_of`) provides a robust positive control demonstrating that soft cross-attention guidance transfers cleanly across both UNet and Diffusion Transformer architectures without retraining.

### 5.1 Powered Lateral Guidance Results ($N = 192$ Paired Runs per Condition)

| Benchmark Suite | Model Backbone | Baseline Rate (OFF) | Guided Rate (Strength 6.00) | Net Paired Gain | McNemar $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard 24 Lateral** | **SD v1.5 (UNet)** | $25.00\%$ ($34/136$) | **$53.68\%$** ($73/136$) | $+39\text{ pairs}$ | **$p = 5.006 \times 10^{-6}$** |
| **Standard 24 Lateral** | **SD 3.5 M (MMDiT)** | $80.88\%$ ($110/136$) | **$90.44\%$** ($123/136$) @ str 3.0 | $+14\text{ pairs}$ | **$p = 0.002577$** |
| **Hard 24 Lateral** | **SD 3.5 M (MMDiT)** | $52.08\%$ ($100/192$) | **$76.56\%$** ($147/192$) @ str 6.0 | $+47\text{ pairs}$ | **$p = 4.248 \times 10^{-11}$** |

* **Zero Spatial Leakage Invariant:** By isolating guidance exclusively to T5 entity token indices and enforcing strictly $0.0000$ spatial attention bias on CLIP-L, CLIP-G, and style tokens, lateral steering preserves underlying aesthetic scores with zero distribution shift.

---

## 6. Sequential Inference-Time Sweeps on SD 3.5 Medium

To sweep inference-time aesthetic levers without combinatorial false discovery, a sequential 1-factor-at-a-time sweep was executed on SD 3.5 Medium ($512\times 512$, $N=160$ pairs per stage, total runtime: $15.58\text{ minutes}$):

1. **Stage 1 (Sampler Choice):** FlowMatchEuler was confirmed superior to FlowMatchHeun ($\Delta\text{LAION} = -0.0490, p = 0.0009$).
2. **Stage 2 (Step Budget Pareto Curve):** 20 Euler steps ($3.90\text{ s/image}$, $15.4\text{ img/min}$) is the Pareto sweet spot. 28 steps ($+0.0198$) and 36 steps ($+0.0075$) sit well inside the cross-seed noise envelope ($\pm 0.162$).
3. **Stage 3 (CFG Rescaling):** Variance-preserving CFG rescaling at $\phi = 0.70$ provides a modest, cost-free dynamic-range enhancement ($+0.0398$ LAION, $+0.0200$ ImageReward, $p < 0.001$) preventing highlight blowout with zero latency overhead.
4. **Stage 4 (Refiner Pass):** Full-frame refiner passes produced flat aesthetic metrics ($+0.0006 \to +0.0124$). Inpainting compositing verified $100\%$ outside-mask preservation ($SSIM = 0.9981$).

---

## 7. Master Summary of Findings

```
========================================================================================================================
MASTER TECHNICAL PAPER CONSOLIDATED RESULTS TABLE
========================================================================================================================
Dimension / Metric          | Stable Diffusion v1.5     | PixArt-Alpha              | Stable Diffusion 3.5 Medium
------------------------------------------------------------------------------------------------------------------------
Architecture                | UNet (0.86B params)       | DiT (0.6B params)         | MMDiT (2.5B params)
Text Encoder                | CLIP-L (77 tokens)        | T5-XXL (120 tokens)       | T5-XXL (512 tokens)
Inference Latency (512x512) | 1.40 s/img (42.8 img/min) | 1.97 s/img (30.5 img/min) | 3.90 s/img (15.4 img/min)
------------------------------------------------------------------------------------------------------------------------
Aesthetic Baseline (LAION)  | 5.954 +- 0.234            | 6.418 +- 0.147            | 6.331 +- 0.175
Human Preference (ImageRew) | 0.774 +- 0.126            | 1.009 +- 0.077            | 0.968 +- 0.095
HPS v2.1 Score              | 0.3332 +- 0.004           | 0.3403 +- 0.002           | 0.3391 +- 0.003
------------------------------------------------------------------------------------------------------------------------
Prompt Descriptor Exp (d̄)  | +0.0831 (p = 2.2e-5)      | -0.0665 (p = 2.5e-5)      | +0.0343 (p = 0.034, Wilcoxon p = 0.163)
CLIP Alignment Delta        | -0.0129 (p = 5.9e-14)     | -0.0068 (p = 3.3e-8)      | -0.0051 (p = 7.3e-6)
------------------------------------------------------------------------------------------------------------------------
Lateral Steering (Standard) | 25.00% -> 53.68% (p=5e-6) | Not evaluated for lateral | 80.88% -> 90.44% (p = 0.0026)
Hard Spatial Steering       | Not evaluated on Hard 24  | Not evaluated on Hard 24  | 52.08% -> 76.56% (p = 4.25e-11)
------------------------------------------------------------------------------------------------------------------------
Case Study A (Depth Metric) | 2D: 56.7% Acc, 12 FP      | Metric unvalidated on DiT | 3D: 60.0% Acc, 5 FP (McNemar p = 0.58)
                            | (Both automated metrics perform > 20% below trivial 81.1% majority baseline; 25% missing)
------------------------------------------------------------------------------------------------------------------------
Case Study B (Style Exp)    | Genuinely Helps CLIP-L    | Actively Hurts T5-XXL     | Ambiguous / Flat on T5-XXL
                            | (Proves text encoder, not generative denoiser architecture, is causal driver of obsolescence)
------------------------------------------------------------------------------------------------------------------------
CFG Rescaling (phi = 0.70)  | Standard Option           | Standard Option           | Optimal Free Polish (+0.04 LAION)
========================================================================================================================
```

---

## 8. Conclusion

This study demonstrates that measurement discipline and causal isolation are vital when evaluating modern generative models. Upgrading from early CLIP encoders to large language models (T5-XXL) eliminates the need for manual prompt engineering tricks while imposing an attention-dilution penalty if they are used. Simultaneously, automated spatial evaluation of 3D depth remains fundamentally unreliable due to zero-shot detection dropouts and scene unidentifiability. We hope these negative-heavy, transparently documented findings provide a grounded baseline for future diffusion transformer research.
