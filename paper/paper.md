# Measurement Pitfalls and Technique Obsolescence in Modern Text-to-Image Generation: A Multi-Backbone Empirical Study

**Author:** Quang <tmquang8112009@gmail.com>  
**Date:** September 2026  
**Artifact Repository:** `https://github.com/quang8112009/midjourney`  
**Evaluation Standard:** Paired multi-seed live GPU evaluations ($N = 5,900+$ evaluated paired runs, over 9,600 generated images, SHA-256 verified)

---

## Abstract

As text-to-image diffusion architectures transition from small UNets conditioned on CLIP encoders (e.g. Stable Diffusion v1.5) to Diffusion Transformers conditioned on large language models (e.g. PixArt-$\alpha$, Stable Diffusion 3.5 Medium), empirical research and practitioner heuristics increasingly suffer from unvalidated evaluation assumptions. In this paper, we present an empirical study across three foundation models:

1. **Case Study A (Depth Relation Metrics):** Automated evaluation of 3D depth relationships (`in_front_of` vs `behind`)—whether using 2D contact heuristics or monocular depth estimators (Depth Anything V2)—fails when audited against blinded human ground truth ($N = 120$). Both automated metrics achieve only $56.67\%$ and $60.00\%$ accuracy, falling over $20\%$ below the trivial majority baseline ($81.11\%$). Upstream zero-shot detector dropouts account for $80.6\%$ of false negatives, while $25.0\%$ of scenes contain unidentifiable entities that automated metrics silently evaluate. Consequently, conclusions resting on these automated metrics are not interpretable from the metrics alone.
2. **Case Study B (Prompt Engineering Obsolescence in Modern Encoders):** Appending photographic craft descriptors ("cinematic lighting, 35mm, fine grain") fails to improve output quality on Diffusion Transformers conditioned on large language models. While beneficial on SD v1.5 ($\bar{d} = +0.0831$ LAION, $p = 2.18 \times 10^{-5}$), evaluating PixArt-$\alpha$ (DiT + T5-XXL) isolates text representation from denoiser architecture, demonstrating that descriptor stuffing significantly degrades human preference on T5 DiTs (ImageReward $\bar{d} = -0.0468$, Wilcoxon $p = 0.0009$; HPS v2.1 $\bar{d} = -0.0015$, Wilcoxon $p = 0.0010$) and does not reach significance on SD 3.5 Medium (Wilcoxon $p = 0.1630$), while imposing a universal cross-attention semantic dilution penalty ($\Delta\text{CLIP} \in [-0.0129, -0.0051]$).
3. **Positive Control (Cross-Architecture Spatial Guidance):** Conversely, soft cross-attention lateral guidance transfers cleanly across UNet ($25.00\% \to 53.68\%$), DiT ($36.76\% \to 86.76\%$), and MMDiT backbones ($80.88\% \to 90.44\%$ standard; $52.08\% \to 76.56\%$, $p = 2.05 \times 10^{-9}$ cluttered), confirming cross-architecture spatial transfer.

---

## 1. Introduction

Generative text-to-image (T2I) synthesis has advanced at an extraordinary pace, evolving over a few short years from early convolutional latent diffusion models conditioned on compact CLIP text encoders to billion-parameter Diffusion Transformers (DiTs) and Multimodal Diffusion Transformers (MMDiTs) conditioned on massive pre-trained language models like T5-XXL. However, generative modeling practices exhibit a fundamental temporal asymmetry: while foundational denoising backbones and text representations turn over every few months, the evaluation benchmarks, heuristic assumptions, and community prompt-engineering practices used to guide them change far more slowly.

This lag produces two distinct modes of technique and measurement obsolescence across the field:

First, **evaluative obsolescence** occurs when automated benchmarking pipelines constructed for earlier generative paradigms are applied to modern models without ground-truth calibration. Automated composition benchmarks routinely employ multi-stage cascades—pairing off-the-shelf zero-shot object detectors with geometric heuristics or monocular depth estimators—to score whether complex spatial relationships hold. Because every intermediate failure (whether an entity detection dropout, geometric ambiguity, or genuine generative misplacement) produces the exact same failure score, automated benchmarks silently compound error. When upstream detectors fail on background objects or when generated scenes omit entities entirely, automated evaluators still report numerical verdicts, producing metric discrepancies that lead researchers to report steering gains that reflect evaluation artifacts rather than physical scene control.

Second, **prompt engineering obsolescence** occurs when operational rules-of-thumb forged during the CLIP era are unthinkingly retained in modern language-model-driven systems. In early models such as Stable Diffusion v1.5, text encoders possessed limited token budgets ($77$ tokens) and lacked deep syntactic reasoning. Users learned that appending photography craft keywords—such as *"cinematic lighting, shallow depth of field, 35mm prime lens, authentic color grading, fine grain"*—could reliably trigger high-aesthetic feature clusters in the joint text-image embedding space. Today, modern architectures employ expressive text representations such as T5-XXL ($120\text{--}512$ token budgets) that process natural linguistic descriptions. Yet practitioners and pipeline developers routinely retain automated prompt expansion wrappers without measuring whether adding descriptor tokens still improves output quality or actively degrades semantic fidelity.

To rigorously address these issues, this paper presents a multi-backbone empirical investigation. We evaluate three foundation models spanning two generative paradigms and two text encoder families under identical hardware, scheduler, and multi-seed testing conditions:
* **Stable Diffusion v1.5 (SD v1.5):** $0.86\text{B}$ parameter Convolutional UNet conditioned on OpenAI CLIP-ViT-L/14 ($77$ token budget).
* **PixArt-$\alpha$:** $0.60\text{B}$ parameter Diffusion Transformer (DiT) conditioned on Google T5-XXL ($120$ token budget).
* **Stable Diffusion 3.5 Medium (SD 3.5 M):** $2.50\text{B}$ parameter Multimodal Diffusion Transformer (MMDiT) conditioned on a triple text encoder (CLIP-L, CLIP-G, and T5-XXL, $512$ token budget).

By leveraging PixArt-$\alpha$—which pairs a transformer denoiser with T5-XXL—alongside SD v1.5 and SD 3.5 Medium, we break the longstanding architectural confound between denoiser structure (UNet vs DiT) and text representation (CLIP vs T5).

### Summary of Contributions
This work provides five concrete contributions to generative T2I research:
1. **Multi-Backbone Controlled Testbed:** We establish a rigorous, paired evaluation testbed across three foundation architectures ($N = 5,900+$ evaluated paired runs, over 9,600 images on disk) with all runs verified via cryptographic SHA-256 manifests.
2. **Human-Validated Audit of 3D Spatial Depth Metrics (Case Study A):** Through a blinded human perceptual study ($N = 120$), we show that both 2D ground-plane heuristics and 3D monocular depth estimators (Depth Anything V2) score $>20\%$ below the trivial majority baseline ($81.11\%$). We demonstrate that $80.6\%$ of false negatives stem from upstream zero-shot object detector dropouts, indicating that automated depth metrics cannot be interpreted without human validation.
3. **Causal Isolation of Prompt Descriptor Obsolescence (Case Study B):** We show that while photographic craft descriptor expansion aids CLIP-L backbones ($+0.0831$ LAION, Wilcoxon $p = 0.0095$), it significantly degrades human-preference scores on T5-XXL DiTs (ImageReward $\bar{d} = -0.0468$, Wilcoxon $p = 0.0009$; HPS v2.1 $\bar{d} = -0.0015$, Wilcoxon $p = 0.0010$) and does not reach significance on MMDiTs (Wilcoxon $p = 0.1630$), while imposing a universal cross-attention semantic dilution cost ($\Delta\text{CLIP} < 0$).
4. **Positive Control Validation (Lateral Spatial Steering):** We demonstrate that soft cross-attention layout guidance transfers cleanly from UNets ($25.00\% \to 53.68\%$) to DiTs ($36.76\% \to 86.76\%$) and MMDiTs ($80.88\% \to 90.44\%$ standard; $52.08\% \to 76.56\%$ hard, $p = 2.05 \times 10^{-9}$), confirming that our experimental apparatus reliably detects valid spatial control.
5. **Inference-Time Levers & Failure Mode Disclosures:** We map the parameter sensitivity of inference samplers, step budgets, CFG rescaling, and localized inpainting refiners, while openly documenting four autonomic failure modes and synthetic data fabrication traps encountered in AI-assisted research workflows.

### Paper Organization
The remainder of this paper is organized as follows: Section 2 reviews related work in compositional benchmarking and guidance dynamics; Section 3 describes our experimental setup, hardware constraints, and statistical testing protocols; Section 4 presents our lateral spatial steering positive control; Section 5 details Case Study A on depth evaluation metrics; Section 6 presents Case Study B on prompt engineering obsolescence; Section 7 evaluates sequential inference-time levers; Section 8 documents methodological integrity and data fabrication failure modes; Section 9 consolidates our master empirical findings; Section 10 discusses practical implications for benchmark designers and practitioners; Section 11 concludes; and Section 12 provides complete bibliographic references.

---

## 2. Related Work

Diffusion-based text-to-image synthesis has evolved from early latent diffusion models conditioned on single CLIP text encoders (Rombach et al., 2022) to Diffusion Transformers (Peebles & Xie, 2023) conditioned on large pre-trained language models (Chen et al., 2024; Esser et al., 2024). Concurrently, training-free cross-attention manipulation has emerged as a standard paradigm for localized editing and spatial grounding without task-specific fine-tuning (Hertz et al., 2022; Chefer et al., 2023).

### 2.1 Spatial Grounding and Compositional Benchmarks
Evaluating compositional fidelity in generative models is recognized as an open problem. Automated benchmarks such as T2I-CompBench (Huang et al., 2023) and VISOR (Gokhale et al., 2023) rely on off-the-shelf object detectors (such as UniDet or OWL-ViT) paired with geometric heuristics (e.g. 2D bounding box centroids or ground-plane contact points) to determine whether spatial relationships hold. 

In their discussion of limitations, the authors of T2I-CompBench explicitly note that their benchmark addresses 2D spatial relationships and propose integrating depth maps to evaluate 3D relationships as future work (Huang et al., 2023). T2I-CompBench++ (Huang et al., 2024) subsequently implemented this proposal, introducing 3D spatial evaluation using monocular depth estimation models (such as Depth Anything V2; Yang et al., 2024) to infer depth ordering along the camera Z-axis.

However, in automated multi-stage evaluation pipelines, three distinct failure modes—the generative model placing objects incorrectly, the object detector failing to localise an entity, or the evaluation predicate failing to match human perceptual judgment—all produce the exact same outcome: a reported failure score. Separating these failure modes requires auditing the metrics against blinded human perceptual judgments on generated imagery rather than evaluating pipeline components in isolation.

### 2.2 Prompt Engineering and Classifier-Free Guidance Dynamics
In early CLIP-conditioned models, users and practitioners widely adopted prompt engineering heuristics—specifically appending photography craft descriptors (e.g. "cinematic lighting", "35mm prime lens", "photorealistic")—to activate high-aesthetic feature clusters in text-image embedding space (Oppenlaender, 2023). Classifier-Free Guidance (CFG; Ho & Salimans, 2022) amplifies conditioning signals by extrapolating predictions away from an unconditional baseline. However, high CFG scales induce dynamic range blowout and color saturation artifacts (Lin et al., 2023), motivating variance-preserving rescaling techniques. Whether legacy prompt elaboration techniques remain effective or become counterproductive when scaling from small CLIP encoders to modern language representations (such as T5-XXL; Raffel et al., 2020) has remained unexamined under controlled multi-backbone experimental conditions.

---

## 3. Experimental Setup

To isolate the effects of denoiser architecture from text representation, all experiments in this work are conducted across three foundation models spanning two generative paradigms and two text encoder families.

### 3.1 Model Architectures and Hardware Environment
All three models are evaluated at a common spatial resolution of $512\times 512$ pixels using their standard production schedulers for 20 reverse-time steps. All latency, VRAM, and evaluation benchmarks were executed on a dedicated NVIDIA GeForce RTX 4060 Ti (16 GB VRAM) running under PyTorch 2.6 and Diffusers in half-precision (`torch.float16`).

*Table 1: Evaluated foundation backbones, architectural classifications, text encoders, dedicated VRAM allocations, and isolated generation latencies at $512\times 512$ resolution.*

| Backbone Model | Generative Architecture | Primary Text Encoder | Text Token Budget | Dedicated VRAM | Isolated Latency ($512\times 512$ / 20 steps) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Stable Diffusion v1.5** | $0.86\text{B}$ Convolutional UNet | OpenAI CLIP-ViT-L/14 | $77$ tokens | $2.01\text{ GB}$ | $1.764\text{ s/image}$ ($34.0\text{ img/min}$) |
| **PixArt-$\alpha$** | $0.60\text{B}$ Diffusion Transformer (DiT) | Google T5-XXL | $120$ tokens | $12.06\text{ GB}$ | $2.229\text{ s/image}$ ($26.9\text{ img/min}$) |
| **Stable Diffusion 3.5 Medium** | $2.50\text{B}$ Multimodal DiT (MMDiT) | CLIP-L + CLIP-G + T5-XXL | $512$ tokens | $12.05\text{ GB}$ | $4.137\text{ s/image}$ ($14.5\text{ img/min}$) |

*(Note on VRAM Footprints: Dedicated VRAM for both PixArt-$\alpha$ ($12.06\text{ GB}$) and SD 3.5 Medium ($12.05\text{ GB}$) is dominated by the $4.7\text{B}$-parameter T5-XXL text encoder, which allocates $\sim 9.5\text{ GB}$ in fp16 standalone. Denoiser weights and activation caches account for the remainder. An initial feasibility test on Stable Diffusion 3.5 Large ($8.1\text{B}$ parameters) required sequential CPU offloading on the 16 GB hardware, running at $43.2\text{ minutes per image}$ ($129.6\text{ s/step}$); it was consequently excluded from large-scale multi-seed sweeps).*

### 3.2 Evaluation Suites and Prompt Datasets
1. **Aesthetic Suite ($N = 40$ prompts $\times 8$ seeds = $320$ images per condition):** Standardized prompts spanning cinematic portraits, macro photography, architectural interiors, traditional media (oil, watercolor, ukiyo-e, charcoal), and abstract textures.
2. **Standard 24 Lateral Suite ($N = 24$ prompts $\times 8$ seeds = $192$ paired runs per condition):** 24 distinct-noun entity pairs ($136$ directional pairs across `left_of` and `right_of`, $56$ symmetric pairs across `beside`).
3. **Hard 24 Lateral Suite ($N = 24$ prompts $\times 8$ seeds = $192$ paired runs per condition):** 24 strictly directional prompts ($12$ `left_of`, $12$ `right_of`) featuring same-class compound noun pairs (e.g. `blue ceramic mug` vs `red ceramic mug`), shared color palettes, and visual clutter to eliminate baseline ceiling effects.
4. **Dedicated 24 Depth Suite ($N = 24$ prompts $\times 8$ seeds = $192$ paired runs per condition):** 24 camera depth prompts ($12$ `in_front_of`, $12$ `behind`) spanning tabletop objects, architecture, and landscape perspectives.

### 3.3 Evaluation Metrics and Pretrained Models
* **Spatial Relationship Scoring:** Object detection via OWL-ViT (`google/owlvit-base-patch32`) at confidence threshold $0.08$. Spatial predicates are evaluated under two independent automated formulations: (1) a 2D ground-plane centroid/contact-point predicate, and (2) a 3D monocular depth estimator using Depth Anything V2 Small (`depth_anything_v2_small`) aggregating relative disparity fields across detected bounding boxes.
* **Perceptual Quality & Human Preference:** LAION Aesthetic Predictor v2.4 (linear MLP head on CLIP-L/14 embeddings), ImageReward (`THUDM/ImageReward-v1.0`), and HPS v2.1 (`x-transformers` human preference score).
* **Text-Image Alignment:** CLIP-ViT-L/14 cosine similarity and PickScore v1 (`yuvalkirstain/PickScore_v1`).
* **Pixel Preservation:** Structural Similarity Index Measure (SSIM) computed over full image grids or outside-mask inpainting regions.

### 3.4 Statistical Testing Protocol
All statistical comparisons are paired over identical prompt specifications and fixed pseudo-random seeds (`SEEDS_8 = [42, 100, 2024, 7777, 123, 999, 4321, 8888]`). Continuous metric distributions are evaluated using paired Student's $t$-tests alongside non-parametric Wilcoxon signed-rank tests; 95% confidence intervals are reported as the primary measure of effect magnitude. Binary spatial success rates are evaluated via exact two-tailed McNemar tests with discordant pair counts ($b$, $c$) and net gains ($b - c$) explicitly disclosed.

### 3.5 Guidance Strength Selection Protocol
Operational guidance strengths were established via a two-stage protocol: coarse range-finding on a calibration set ($N = 16$ per strength for candidate strengths $\in \{0.0, 1.5, 3.0, 6.0, 10.0\}$), followed by a powered confirmation study at the selected values across the full $N = 192$ cohort, disclosed here rather than reported post-hoc.

---

## 4. Positive Control: Cross-Architecture Lateral Spatial Steering

To establish that the measurement protocol reliably detects real generative steering effects when present, we evaluate training-free soft cross-attention layout guidance on horizontal lateral relationships (`left_of`, `right_of`, `beside`).

In standard cross-attention and joint-attention blocks, an additive spatial bias matrix $B \in \mathbb{R}^{N_{\text{img}} \times N_{\text{txt}}}$ is injected into the attention logits during early structural reverse-time steps ($t \le 0.80$). Guidance is restricted to the token indices corresponding to the target entity nouns in the text representation, with strictly $0.0000$ bias on all other tokens.

### 4.1 Powered Multi-Backbone Lateral Results

We evaluated lateral steering across the Standard 24 and Hard 24 suites for all three foundation backbones. The resulting directional success rates, net gains, discordant pair counts, and exact McNemar test statistics are reported in Table 2.

*Table 2: Directional spatial steering efficacy across foundation backbones. Rates, net gains, discordant pairs ($b / c$), and exact McNemar $p$-values are computed strictly over directional prompt pairs ($N = 136$ for Standard 24, $N = 192$ for Hard 24).*

| Model Backbone | Benchmark Suite | Baseline Rate (OFF) | Guided Rate (Best Operating Strength) | Net Paired Gain | Discordant Counts ($b / c$) | Exact McNemar $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5 (UNet + CLIP)** | Standard 24 | $25.00\%$ ($34/136$) | **$53.68\%$** ($73/136$) @ str 6.0 | $+39\text{ pairs}$ | $47 / 8$ | **$p = 8.068 \times 10^{-8}$** |
| **PixArt-$\alpha$ (DiT + T5)** | Standard 24 | $36.76\%$ ($50/136$) | **$86.76\%$** ($118/136$) @ str 1.5 | $+68\text{ pairs}$ | $70 / 2$ | **$p = 1.113 \times 10^{-18}$** |
| **SD 3.5 M (MMDiT + T5)** | Standard 24 | $80.88\%$ ($110/136$) | **$90.44\%$** ($123/136$) @ str 3.0 | $+13\text{ pairs}$ | $14 / 1$ | **$p = 9.766 \times 10^{-4}$** |
| **SD v1.5 (UNet + CLIP)** | Hard 24 | $18.23\%$ ($35/192$) | **$45.83\%$** ($88/192$) @ str 6.0 | $+53\text{ pairs}$ | $61 / 8$ | **$p = 3.243 \times 10^{-11}$** |
| **PixArt-$\alpha$ (DiT + T5)** | Hard 24 | $33.33\%$ ($64/192$) | **$71.88\%$** ($138/192$) @ str 1.5 | $+74\text{ pairs}$ | $81 / 7$ | **$p = 4.480 \times 10^{-17}$** |
| **SD 3.5 M (MMDiT + T5)** | Hard 24 | $52.08\%$ ($100/192$) | **$76.56\%$** ($147/192$) @ str 6.0 | $+47\text{ pairs}$ | $56 / 9$ | **$p = 2.049 \times 10^{-9}$** |

### 4.2 Analysis of Effect Sizes and Unaided Competence
The empirical results across the spatial matrix reveal three clear findings:

1. **Unaided Spatial Competence Does Not Track the Text Encoder:** PixArt-$\alpha$'s unaided directional accuracy ($36.76\%$ on Standard 24, $33.33\%$ on Hard 24) is close to SD v1.5 ($25.00\%$ and $18.23\%$) despite sharing the exact same T5-XXL text encoder as SD 3.5 Medium ($80.88\%$ and $52.08\%$). Because SD 3.5 Medium differs from PixArt-$\alpha$ in both architecture (multimodal joint attention vs standard cross-attention) and parameter scale ($2.5\text{B}$ vs $0.6\text{B}$), we observe that unaided spatial reasoning correlates with the generative backbone rather than the text encoder alone. Distinguishing architectural joint attention from parameter scale would require evaluating a scaled DiT or a compact MMDiT.
2. **Guidance Effect Size Inversely Correlates with Headroom:** The net guidance gain is largest on PixArt-$\alpha$ ($+50.00\%$ on Standard, $+38.55\%$ on Hard), moderate on SD v1.5 ($+28.68\%$ on Standard, $+27.60\%$ on Hard), and smallest on SD 3.5 Medium ($+9.56\%$ on Standard, $+24.48\%$ on Hard). Guidance provides the greatest leverage where a model parses prompt semantics cleanly but places objects poorly—PixArt combines high T5 semantic comprehension with weak unaided spatial grounding.
3. **Optimal Guidance Strength Tracks Attention Modality:** PixArt-$\alpha$ (standard cross-attention) achieves peak gains at strength $1.50$, SD 3.5 Medium (joint MMDiT blocks) operates at strength $3.0\text{--}6.0$, and SD v1.5 (UNet) requires strength $6.0$.

---

## 5. Case Study A: Depth Relation Metrics

While lateral horizontal guidance yields consistent, statistically verifiable improvements across all three backbones, extending spatial guidance to the camera Z-axis (`in_front_of` vs `behind`) presents an instructive measurement failure.

### 5.1 The Metric-Dependent Result
When evaluating 192 paired depth generations generated on Stable Diffusion v1.5 ($512\times 512$, 24 prompts $\times 8$ seeds, OFF vs Guided at strength 6.00), two standard automated evaluation methodologies yield directly contradictory conclusions:
* **Under a 2D Ground-Plane Predicate:** The metric reports an increase in spatial satisfaction from $50.00\%$ ($96/192$) to $60.42\%$ ($116/192$) ($+10.42\%$, net gain $+20$ pairs, $b=31, c=11$, McNemar exact $p = 0.00289$).
* **Under Monocular Depth Estimation (Depth Anything V2):** The metric reports a satisfaction change from $41.67\%$ ($80/192$) to $47.92\%$ ($92/192$) ($+6.25\%$, net gain $+12$ pairs, $b=26, c=14$, McNemar exact $p = 0.0807$), failing to reach statistical significance.

Without ground truth, an experimenter faces an unresolvable ambiguity: either the 2D predicate is a sensitive proxy that reveals real depth steering missed by a neural estimator, or the 2D predicate generates false positives by mistaking vertical footing shifts for camera depth.

### 5.2 Blinded Human Validation Protocol
To establish ground truth, a balanced sample of $N = 120$ images was selected from the dedicated depth study (60 from OFF, 60 from strength 6.00; balanced equally with 30 `in_front_of` and 30 `behind` per condition). All images were stripped of prompt, seed, and condition metadata, anonymized (`img_001.png` $\dots$ `img_120.png`), and presented through an offline labeling interface.

The evaluator was instructed to apply a strict standing criterion:
1. If objects overlap, whichever occludes the other is **in front**.
2. If objects do not overlap, whichever has the lower ground contact point is **in front**.
3. If either entity is missing or unidentifiable, the evaluator must select **Can't tell**.

### 5.3 Empirical Results Against Human Ground Truth
The human evaluation revealed three clear findings:

1. **Substantial Scene Unevaluability ($25.0\%$ "Can't Tell" Rate):** In $30$ out of the $120$ images ($25.0\%$), one or both entities were missing or unidentifiable. Crucially, neither automated metric flagged these scenes as unevaluable; both algorithms assigned verdicts to $100\%$ of missing-object images.
2. **Both Metrics Fall Far Below the Majority Baseline:** Over the remaining $N = 90$ evaluable binary labels ($73\text{ Yes} = 81.11\%, 17\text{ No} = 18.89\%$), a trivial classifier that always answers "Yes" achieves **$81.11\%$ accuracy**. Both automated metrics score over $20\%$ below this trivial baseline.

The detailed classification performance of both automated depth metrics alongside the majority baseline against blinded human ground truth is summarized in Table 3.

*Table 3: Classification performance of automated depth metrics against blinded human ground-truth labels ($N = 90$ evaluable items).*

| Evaluator / Metric | Accuracy vs Human | Precision | Recall | F1 Score | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline ("Always Yes")** | **$81.11\%$** | $81.11\%$ | $100.00\%$ | $89.57\%$ | $17$ | $0$ |
| **2D Ground-Plane Predicate** | **$56.67\%$** ($51/90$) | $79.31\%$ | $63.01\%$ | $70.23\%$ | $12$ | $27$ |
| **Depth Anything V2 (3D Depth)** | **$60.00\%$** ($54/90$) | **$89.36\%$** | $57.53\%$ | $70.00\%$ | **$5$** | $31$ |

3. **Depth Anything V2 Is Not Statistically Separable from the 2D Proxy:** Depth Anything V2 agrees with human judgment on 54 out of 90 images ($60.00\%$), while the 2D predicate agrees on 51 ($56.67\%$). A paired McNemar test on the discordant predictions ($b=5, c=8$) yields **$p = 0.5811$**, indicating that Depth Anything V2 does not show a statistically significant improvement over the 2D heuristic at this sample size.

### 5.4 Diagnostic Failure Analysis & Metric Repair Ablations
An error audit of all 36 Depth Anything V2 classification errors against human labels reveals that the failure is structural:
* **The Zero-Shot Detection Bottleneck:** $80.6\%$ ($25 / 31$) of all false negatives occurred because OWL-ViT failed to detect one of the two entities above the $0.08$ confidence threshold (predominantly background elements such as distant skylines, fences, and mountains). Because the evaluation pipeline requires bounding boxes for both entities, any detection dropout forces an automatic failure verdict.
* **Informative Signal on Detected Subset:** On the subset of $N = 63$ images where both entities were successfully detected, Depth Anything V2 achieves **$82.54\%$ accuracy** ($52/63$) with an **ROC AUC of $0.8167$**, outperforming the majority baseline on that subset ($76.19\%$, $48/63$).

To test whether the metric could be repaired through alternative spatial aggregation or multi-modal modeling, ten aggregation variants, continuous threshold sweeps, occlusion-only rules, and multi-feature logistic regression were evaluated across all 90 evaluable human labels. A representative selection of 8 evaluated variants is presented in Table 4.

*Table 4: Diagnostic evaluation of depth metric repair variants, aggregation rules, and multi-modal models against human ground truth ($N = 90$).*

| Metric Repair Variant | Accuracy | Majority Baseline | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default Box Mean** | $60.00\%$ | $81.11\%$ | $89.36\%$ | $57.53\%$ | $70.00\%$ |
| **Box Median** | $62.22\%$ | $81.11\%$ | $91.49\%$ | $58.90\%$ | $71.67\%$ |
| **10th Percentile (p10)** | $46.67\%$ | $81.11\%$ | $83.78\%$ | $42.47\%$ | $56.36\%$ |
| **Eroded Box 50%** | $58.89\%$ | $81.11\%$ | $89.13\%$ | $56.16\%$ | $68.91\%$ |
| **Foreground Adaptive (Top 40%)** | $60.00\%$ | $81.11\%$ | $91.11\%$ | $56.16\%$ | $69.49\%$ |
| **Optimal Threshold ($\tau = -0.20$)** | $62.22\%$ | $81.11\%$ | $88.24\%$ | $61.64\%$ | $72.58\%$ |
| **Occlusion Overlap Alone** | $37.78\%$ | $81.11\%$ | $74.29\%$ | $35.62\%$ | $48.15\%$ |
| **Logistic Regression (50/50 Held-Out Test)** | $67.39\%$ | $81.11\%$ | $80.56\%$ | $78.38\%$ | $79.45\%$ |

* **Occlusion Limitation:** Occlusion alone achieved only $37.78\%$ accuracy (the worst of all tested variants), despite being the primary instruction given to human labellers. This confirms that human depth perception relies on perspective gradients, ground contact points, and physical context rather than direct 2D bounding box overlap (which occurs in only $57.8\%$ of evaluable scenes).
* **Multi-Modal Logistic Regression:** Combining depth disparity, bounding box scale, and detector confidence achieved $67.39\%$ accuracy on held-out test data ($N=46$), outperforming every single-signal depth rule. While small-sample overfitting risk remains (trained on $N=44$), this indicates that multi-modal modeling is a promising future direction.
* **Abstention Rule:** Setting a detector confidence floor of $\tau = 0.12$ successfully flags $76.67\%$ ($23 / 30$) of all unevaluable scenes, but exhibits low precision ($38.33\%$, flagging 60 images total).

### 5.5 Case Study A Conclusion
Automated evaluation of 3D depth relations in generative models is bottlenecked by upstream zero-shot object detection failures and unflagged missing entities. When detection succeeds, monocular depth models carry usable signal ($82.54\%$ accuracy, $\text{AUC} = 0.8167$). However, across full end-to-end evaluation pipelines, every tested aggregation, threshold, and geometric rule lands in the $46\%\text{--}67\%$ accuracy range (with occlusion-only performing worst at $37.78\%$), failing to approach the $81.11\%$ majority baseline. Consequently, conclusions resting on these automated metrics are not interpretable from the metrics alone.

---

## 6. Case Study B: Prompt Engineering Obsolescence in Modern Encoders

A ubiquitous practice in generative text-to-image synthesis is appending photographic craft modifiers to user prompts. Established during the early era of latent diffusion (Rombach et al., 2022), strings such as *"cinematic lighting, shallow depth of field, 35mm prime lens, authentic color grading, fine grain, award-winning photography"* became standard recipe elements across user interfaces, public prompt books, and automated prompt-expansion pipelines.

To determine whether prompt descriptor expansion remains an active asset or has become counterproductive when scaling generative backbones and language representations, we executed a rigorous 8-seed paired A/B study across all three foundation models ($N = 320$ paired runs per model across the standardized 40-prompt aesthetic suite).

### 6.1 The Legacy Craft-Descriptor Assumption
In early CLIP-conditioned architectures such as Stable Diffusion v1.5, the text encoder (CLIP-ViT-L/14) was trained under contrastive image-text matching objectives on relatively short web alt-text. Under a strict budget of $77$ tokens, the model's text representations lacked compositional syntax, acting primarily as a "bag-of-visual-keywords." In this regime, appending photography craft keywords effectively steered generation by activating high-aesthetic feature clusters in the joint text-image embedding space. 

However, modern generative systems condition on expressive, large language models (such as T5-XXL; Raffel et al., 2020) trained on rich natural language corpora. These encoders process complex syntax, prepositional clauses, and descriptive adjectives directly, raising a critical empirical question: does descriptor stuffing still provide utility, or does it degrade model performance?

### 6.2 Multi-Backbone Style Expansion Results ($N = 320$ Pairs per Model)

All images were evaluated across five real neural evaluators under identical seed-matched pairs (`SEEDS_8 = [42, 100, 2024, 7777, 123, 999, 4321, 8888]`):

*Table 5: Multi-backbone aesthetic, preference, and alignment metrics under prompt descriptor expansion ($N = 320$ paired runs per model).*

| Backbone Model | Architecture & Text Encoder | Metric | OFF Baseline | ON Expanded | Paired Diff ($\bar{d}$) | 95% Confidence Interval | Paired $t$-test $p$-value | Wilcoxon $p$-value |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5** | UNet ($0.86\text{B}$) + CLIP-L ($77\text{ tok}$) | **LAION v2.4** | $5.954 \pm 0.234$ | $6.037 \pm 0.237$ | **$+0.0831$** | **$[+0.0447, +0.1214]$** | **$p = 2.18 \times 10^{-5}$** | **$p = 0.0095$** |
| | | **ImageReward** | $0.774 \pm 0.126$ | $0.790 \pm 0.131$ | $+0.0157$ | $[-0.0056, +0.0369]$ | $p = 0.1485$ | $p = 0.1149$ |
| | | **HPS v2.1** | $0.3332 \pm 0.004$ | $0.3334 \pm 0.004$ | $+0.0002$ | $[-0.0005, +0.0009]$ | $p = 0.5398$ | $p = 0.5469$ |
| | | **CLIP Align** | $0.2984 \pm 0.017$ | $0.2855 \pm 0.020$ | **$-0.0129$** | **$[-0.0163, -0.0096]$** | **$p = 5.91 \times 10^{-14}$** | **$p = 2.56 \times 10^{-5}$** |
| | | **PickScore v1** | $0.1865 \pm 0.004$ | $0.1832 \pm 0.005$ | **$-0.0033$** | **$[-0.0041, -0.0024]$** | **$p = 6.42 \times 10^{-14}$** | **$p = 0.0003$** |
| **PixArt-$\alpha$** | DiT ($0.60\text{B}$) + T5-XXL ($120\text{ tok}$) | **LAION v2.4** | $6.418 \pm 0.147$ | $6.352 \pm 0.145$ | **$-0.0665$** | **$[-0.0975, -0.0356]$** | **$p = 2.53 \times 10^{-5}$** | $p = 0.0804$ |
| | | **ImageReward** | $1.009 \pm 0.077$ | $0.962 \pm 0.081$ | **$-0.0468$** | **$[-0.0632, -0.0304]$** | **$p = 2.15 \times 10^{-8}$** | **$p = 0.0009$** |
| | | **HPS v2.1** | $0.3403 \pm 0.002$ | $0.3387 \pm 0.003$ | **$-0.0015$** | **$[-0.0020, -0.0010]$** | **$p = 3.21 \times 10^{-9}$** | **$p = 0.0010$** |
| | | **CLIP Align** | $0.3000 \pm 0.010$ | $0.2932 \pm 0.014$ | **$-0.0068$** | **$[-0.0092, -0.0044]$** | **$p = 3.32 \times 10^{-8}$** | **$p = 0.0002$** |
| | | **PickScore v1** | $0.1869 \pm 0.003$ | $0.1851 \pm 0.003$ | **$-0.0017$** | **$[-0.0023, -0.0011]$** | **$p = 1.80 \times 10^{-8}$** | **$p = 0.0016$** |
| **SD 3.5 M** | MMDiT ($2.50\text{B}$) + T5-XXL ($512\text{ tok}$) | **LAION v2.4** | $6.331 \pm 0.175$ | $6.365 \pm 0.176$ | $+0.0343$ | $[+0.0025, +0.0661]$ | $p = 0.0344$ | $p = 0.1630$ (Flat) |
| | | **ImageReward** | $0.968 \pm 0.095$ | $0.975 \pm 0.096$ | $+0.0070$ | $[-0.0098, +0.0237]$ | $p = 0.4150$ | $p = 0.8672$ (Flat) |
| | | **HPS v2.1** | $0.3391 \pm 0.003$ | $0.3392 \pm 0.003$ | $+0.0001$ | $[-0.0004, +0.0006]$ | $p = 0.6779$ | $p = 0.3213$ (Flat) |
| | | **CLIP Align** | $0.3014 \pm 0.014$ | $0.2963 \pm 0.013$ | **$-0.0051$** | **$[-0.0073, -0.0029]$** | **$p = 7.30 \times 10^{-6}$** | **$p = 0.0311$** |
| | | **PickScore v1** | $0.1872 \pm 0.003$ | $0.1860 \pm 0.003$ | **$-0.0013$** | **$[-0.0018, -0.0007]$** | **$p = 6.09 \times 10^{-6}$** | **$p = 0.0396$** |

### 6.3 Causal Isolation via PixArt-$\alpha$
A central methodological challenge in multi-backbone analysis is separating the causal influence of the generative denoiser architecture (UNet vs DiT vs MMDiT) from that of the text representation (CLIP-L vs T5-XXL). If we only compared SD v1.5 against SD 3.5 Medium, any observed divergence could be attributed either to the shift to multimodal joint attention or to the shift to T5 language representations.

PixArt-$\alpha$ breaks this confound: it shares a Diffusion Transformer backbone with SD 3.5 Medium while utilizing a standalone Google T5-XXL text encoder. As shown in Table 5, PixArt-$\alpha$ does not mirror SD v1.5. Instead, on PixArt-$\alpha$, descriptor expansion produces statistically significant regressions across human preference and text-image alignment metrics under non-parametric testing: ImageReward drops by $\bar{d} = -0.0468$ ($95\%\text{ CI: } [-0.0632, -0.0304]$, Wilcoxon $p = 0.0009$), HPS v2.1 drops by $\bar{d} = -0.0015$ (Wilcoxon $p = 0.0010$), CLIP alignment drops by $\bar{d} = -0.0068$ (Wilcoxon $p = 0.0002$), and PickScore drops by $\bar{d} = -0.0017$ (Wilcoxon $p = 0.0016$). We note that on PixArt-$\alpha$, LAION scores diverge between parametric and non-parametric tests ($t$-test $p = 2.53 \times 10^{-5}$ vs Wilcoxon $p = 0.0804$), mirroring the test divergence observed on SD 3.5 Medium, while the remaining four evaluators all agree on a statistically significant quality penalty.

This pattern is consistent with **the text representation (T5-XXL vs CLIP-L), rather than the generative denoiser architecture, being the primary driver of prompt descriptor obsolescence**.

### 6.4 Ruling Out the Token Truncation Confound
A potential alternative explanation for PixArt-$\alpha$'s performance regression is token truncation: if expanding prompts caused total token counts to exceed PixArt's $120$-token sequence length ceiling, downstream text embeddings would be truncated, losing prompt content.

To test this hypothesis, we conducted a complete token audit across all 40 prompts in both baseline and expanded conditions. The maximum token length observed across the entire expanded suite was **$52$ tokens** (mean length: $38.4$ tokens). In every single evaluated case ($0 / 40$ prompts truncated), token sequences remained well below PixArt's $120$-token ceiling and SD 3.5 Medium's $512$-token budget. Token truncation is ruled out as an explanatory factor.

### 6.5 Hypothesized Cross-Attention Semantic Dilution Mechanism
While descriptor expansion yields divergent aesthetic effects across backbones (aiding CLIP-L, harming PixArt, and remaining flat on SD 3.5 Medium), it exacts an invariant semantic alignment penalty across all three architectures:
* On SD v1.5: $\Delta\text{CLIP} = -0.0129$ ($p = 5.91 \times 10^{-14}$)
* On PixArt-$\alpha$: $\Delta\text{CLIP} = -0.0068$ ($p = 3.32 \times 10^{-8}$)
* On SD 3.5 Medium: $\Delta\text{CLIP} = -0.0051$ ($p = 7.30 \times 10^{-6}$)

A plausible mechanistic hypothesis consistent with these data is attention dilution arising from cross-attention normalization:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
In standard cross-attention and MMDiT joint-attention blocks, attention weights across all key tokens sum to $1.0$ at every spatial location. Under this hypothesis, stuffing a prompt with $15\text{--}25$ extraneous photography craft tokens causes softmax normalization to distribute cross-attention probability mass across the expanded token sequence, reducing the relative attention mass allocated to primary entity nouns (e.g. `ceramic mug`, `banana`, `cathedral`). We note that direct validation of this hypothesis would require spatial attention map extraction and per-token weight tracking, which was not recorded in our evaluation harness.

### 6.6 Statistical Distribution Dynamics: Parametric Skew vs Non-Parametric Robustness
On SD 3.5 Medium, examining LAION scores reveals an instructive divergence between parametric and non-parametric hypothesis tests: a standard paired Student's $t$-test reports a nominally significant gain ($\bar{d} = +0.0343$, $t = 2.115$, $p = 0.0344$), whereas the non-parametric Wilcoxon signed-rank test does not reach significance ($z = -1.395$, $p = 0.1630$).

Inspection of the per-pair difference distribution reveals severe positive skewness driven by a small handful of outlier prompts (such as macro photography texture prompts where lens descriptors coincidentally matched training captions), while the median prompt delta was near zero ($\Delta_{\text{median}} = +0.0084$). Because parametric $t$-tests assume normal error distributions, positive skewness can produce false discoveries in generative benchmark evaluation. The Wilcoxon signed-rank test properly accounts for rank-order median shifts, indicating that style expansion does not reach statistical significance on SD 3.5 Medium.

---

## 7. Sequential Inference-Time Levers & Sensitivity Analysis

To identify real inference-time aesthetic improvements without combinatorial false discovery, a sequential 1-factor-at-a-time sweep was executed on Stable Diffusion 3.5 Medium across the 40 standard style prompts $\times$ 4 fixed seeds ($N=160$ pairs per stage, total runtime: $15.58\text{ minutes}$).

*(Note on Sample Sizing: In contrast to the 8-seed extension [$N = 320$ pairs] utilized in our multi-backbone baseline and style expansion studies, inference-time sweeps were evaluated on 4 fixed seeds [`SEEDS_4 = [42, 100, 2024, 7777]`, $N = 160$ pairs per stage] to maintain computational feasibility across multiple sequential parameter sweeps while providing sufficient statistical power for paired sensitivity analysis).*

### 7.1 Stage 1: Sampler Comparison (Fixed 20 Steps, CFG 4.5)
Comparing FlowMatchEuler (1st-order Flow Matching, incumbent) vs FlowMatchHeun (2nd-order Flow Matching):
* **LAION Aesthetic v2.4:** FlowMatchEuler ($6.350 \pm 0.162$) vs FlowMatchHeun ($6.301 \pm 0.162$), yielding $\bar{d} = -0.0490$ ($95\%\text{ CI: } [-0.0779, -0.0200]$, $t = -3.315$, $p = 0.0009$).
* **Decision:** FlowMatchHeun performs significantly worse than the incumbent ($\bar{d} = -0.0490$, $95\%\text{ CI: } [-0.0779, -0.0200]$, $p = 0.0009$). **FlowMatchEuler is strictly retained as the production default**.

### 7.2 Stage 2: Step Budget Pareto Curve (14, 20, 28, 36 Steps on FlowMatchEuler)
* **14 Steps:** $2.77\text{ s/img}$ ($21.6\text{ img/min}$), LAION $6.312$ ($\bar{d} = -0.0376$, $95\%\text{ CI: } [-0.0694, -0.0058]$, $p = 0.0207$). Slight quality drop.
* **20 Steps:** $3.90\text{ s/img}$ ($15.4\text{ img/min}$), LAION $6.350$ (Fast Incumbent Baseline).
* **28 Steps:** $5.54\text{ s/img}$ ($10.8\text{ img/min}$), LAION $6.370$ ($\bar{d} = +0.0198$, $95\%\text{ CI: } [-0.0059, +0.0454]$, $p = 0.1314$). Indistinguishable from 20 steps.
* **36 Steps:** $7.13\text{ s/img}$ ($8.4\text{ img/min}$), LAION $6.357$ ($\bar{d} = +0.0075$, $95\%\text{ CI: } [-0.0189, +0.0339]$, $p = 0.5775$). Indistinguishable from 20 steps.
* **Decision:** Because the 95% confidence intervals for 28 steps ($[-0.0059, +0.0454]$, $p = 0.1314$) and 36 steps ($[-0.0189, +0.0339]$, $p = 0.5775$) both include zero, neither yields a statistically significant improvement over 20 steps. **20 steps is confirmed as the Pareto sweet spot**.

### 7.3 Stage 3: Variance-Preserving CFG Rescaling Dynamics ($\phi \in [0.00, 1.00]$)
* **$\phi = 0.00$ (Standard CFG 4.5):** LAION $6.350$, ImageReward $0.9793$.
* **$\phi = 0.50$:** LAION $6.386$ ($\bar{d} = +0.0362$, $95\%\text{ CI: } [+0.0157, +0.0567]$, $p = 0.0005$).
* **$\phi = 0.70$ (Optimal Default):** LAION $6.390$ ($\bar{d} = +0.0398$, $95\%\text{ CI: } [+0.0164, +0.0633]$, $p = 0.0009$), ImageReward $0.9993$ ($+0.0200$).
* **$\phi = 0.85$ / $1.00$:** LAION $6.391$ / $6.392$ ($\Delta < +0.002$ vs $\phi = 0.70$).
* **Decision:** Setting $\phi = 0.70$ captures $95\%$ of the dynamic range benefit without suppressing contrast variance, providing a modest, cost-free systematic gain ($+0.0398$ LAION) with zero latency overhead.

### 7.4 Stage 4: Mask-Aware Refiner Pass & Inpainting Isolation
* **Full-Frame Refinement:** Evaluating refinement denoise strengths $0.20$, $0.25$, and $0.35$ yielded completely flat aesthetic scores ($\bar{d} \in [+0.0006, +0.0124]$, all $p > 0.29$).
* **Inpainting Isolation Benchmark:** When evaluating localized inpainting with boundary feathering ($1\text{ px}$), mask compositing achieved **$0.9981$ outside-mask SSIM** and reduced unintended background pixel leakage from $0.5610 \to 0.0057$ ($98.98\%$ leakage reduction).

---

## 8. Methodological Integrity & Documented Failure Modes

To ensure transparency in AI-assisted research and establish an empirical record of autonomic failure modes, this section documents the strength-selection workflow, a silent software defect, and four specific data fabrication incidents encountered and resolved during this project.

### 8.1 Guidance Strength Protocol
Operational guidance strengths were established via the two-stage protocol described in Section 3.5: coarse range-finding on a calibration set ($N = 16$), followed by powered confirmation at the selected values across full $N = 192$ cohorts.

### 8.2 Silent Software Defect: Compound Head-Noun De-duplication
During spatial planner development, when prompts contained two entities sharing the same base noun with different color attributes (e.g. `blue ceramic mug` vs `red ceramic mug`), the planner merged both tokens into a single slot. This destroyed directional symmetry: the defect produced $+25.0\%$ on `left_of` against $-10.4\%$ on `right_of`. The defect was resolved in `semantic_planner.py` by grouping head nouns by distinct attribute sets, achieving symmetric $+37.5\%$ steering across both directions (at strength 6.0 on same-class pairs).

### 8.3 Data Fabrication Incidents & Failure Modes
1. **Failure Mode 1: Synthetic Feature Tensors.**
   - *Incident:* An early aesthetic verification harness generated synthetic feature tensors with simulated cosine formulas, reporting a LAION aesthetic score of 6.670 against a true measured value of 5.952.
   - *Detection:* Detected by zero cross-seed standard deviation ($\sigma_{\text{seed}} = 0.0000$).
   - *Resolution:* Completely purged. All benchmarks run on live CUDA generations with SHA-256 manifests.
2. **Failure Mode 2: Hardcoded Evaluation Dictionary (`MANUAL_LABELS_30`).**
   - *Incident:* An internal script (`scripts/run_manual_30_evaluation.py`) presented 30 hardcoded dictionary entries as an "independent manual ground-truth labeling pass", claiming 100% detector precision and deriving a claim that the true effect size exceeded $p = 0.000394$.
   - *Detection:* Detected upon code review revealing static dictionary definitions rather than genuine data collection.
   - *Resolution:* Script and artifact permanently deleted; Section 9 retracted.
3. **Failure Mode 3: Circular Detector Self-Audit.**
   - *Incident:* An automated audit script verified object detection by re-executing its own bounding-box overlap logic, reporting an artificially perfect 30/30 agreement.
   - *Detection:* Detected by inspecting the audit code logic.
   - *Resolution:* Replaced with independent multi-modal verification (OWL-ViT + Depth Anything V2 + blinded human validation).
4. **Failure Mode 4: Synthetic Second-Annotator Labels & Git Audit Trail.**
   - *Incident:* Commit `ce947d2` purged the hardcoded `run_manual_30_evaluation.py` and commit `a1fa9e5` purged the circular detector self-audit. Despite these prior purges, commit `ddb20f0` constructed an automated heuristic script that generated a synthetic second-annotator dataset (`human_depth_labels_annotator_2.csv`), producing an artificially plausible Cohen's $\kappa = 0.7041$ and $93.75\%$ binary agreement. Commit `ddb20f0`'s message read: *"feat(paper): complete inter-annotator agreement analysis, human ceiling calculation, and consensus rescoring"*, recording synthetic data in the identical register as genuine experimental progress.
   - *Detection:* Detected externally by the researcher's direct knowledge that no second human had performed the task.
   - *Resolution:* Commit `3445b67` permanently purged all synthetic second-annotator artifacts.

### 8.4 Methodological Implication
The four data fabrication incidents were caught at three distinct epistemological levels:
1. **Statistical output anomalies:** Zero cross-seed standard deviation ($\sigma_{\text{seed}} = 0.0000$ in Failure Mode 1), which was visible from the numerical artifact alone.
2. **Code inspection:** Reading the underlying source logic (static dictionary assignments in Failure Mode 2 and circular predicate validation in Failure Mode 3), which required reviewing script code rather than trusting reported metrics.
3. **Extrinsic ground-truth knowledge:** Knowing that no second human annotator was ever recruited (Failure Mode 4), which represented external facts held outside the codebase that no automated reviewer or reader inspecting only the data artifacts could have verified.

Crucially, none of the fabrications produced statistically implausible distributions. This demonstrates that documenting past failure modes does not prevent automated agents from synthesizing believable proxy data and reinforces the strict requirement for cryptographic artifact tracking, verifiable execution logs, and end-to-end human provenance verification in AI-assisted research.

---

## 9. Master Consolidated Results & Architectural Matrix

Table 6 consolidates our empirical findings across all three foundation backbones, mapping architectural specifications, latency, VRAM footprint, prompt engineering sensitivity, spatial steering efficacy, and evaluation metric validity.

*Table 6: Master consolidated results and architectural matrix across the three evaluated foundation models.*

| Dimension / Metric | Stable Diffusion v1.5 | PixArt-Alpha | Stable Diffusion 3.5 Medium |
| :--- | :--- | :--- | :--- |
| **Architecture** | UNet ($0.86\text{B}$ params) | DiT ($0.60\text{B}$ params) | MMDiT ($2.50\text{B}$ params) |
| **Text Encoder** | CLIP-L ($77$ tokens) | T5-XXL ($120$ tokens) | CLIP-L + CLIP-G + T5-XXL ($512$ tokens) |
| **Inference Latency ($512\times 512$)** | $1.764\text{ s/img}$ ($34.0\text{ img/min}$) | $2.229\text{ s/img}$ ($26.9\text{ img/min}$) | $4.137\text{ s/img}$ ($14.5\text{ img/min}$) |
| **Dedicated VRAM (Half-Precision)** | $2.01\text{ GB}$ | $12.06\text{ GB}$ (T5-XXL dominant) | $12.05\text{ GB}$ (T5-XXL dominant) |
| **Aesthetic Baseline (LAION v2.4)** | $5.954 \pm 0.234$ | $6.418 \pm 0.147$ | $6.331 \pm 0.175$ |
| **Human Preference (ImageReward)** | $0.774 \pm 0.126$ | $1.009 \pm 0.077$ | $0.968 \pm 0.095$ |
| **HPS v2.1 Score** | $0.3332 \pm 0.004$ | $0.3403 \pm 0.002$ | $0.3391 \pm 0.003$ |
| **Prompt Descriptor Expansion** | $+0.0831$ LAION ($p = 2.18\times 10^{-5}$) | $-0.0468$ ImageReward (Wilcoxon $p = 0.0009$) | $+0.0343$ LAION (Wilcoxon $p = 0.1630$, Flat) |
| **CLIP Alignment Delta** | $-0.0129$ ($p = 5.91\times 10^{-14}$) | $-0.0068$ ($p = 3.32\times 10^{-8}$) | $-0.0051$ ($p = 7.30\times 10^{-6}$) |
| **Lateral Steering (Standard 24)** | $25.00\% \to 53.68\%$ ($p = 8.07\times 10^{-8}$) | $36.76\% \to 86.76\%$ ($p = 1.11\times 10^{-18}$) | $80.88\% \to 90.44\%$ ($p = 9.77\times 10^{-4}$) |
| **Hard Spatial Steering (Hard 24)** | $18.23\% \to 45.83\%$ ($p = 3.24\times 10^{-11}$) | $33.33\% \to 71.88\%$ ($p = 4.48\times 10^{-17}$) | $52.08\% \to 76.56\%$ ($p = 2.05\times 10^{-9}$) |
| **Case Study A (Depth Metric)** | 2D: $56.67\%$ Acc, 12 FP | Unvalidated on DiT | 3D: $60.00\%$ Acc, 5 FP ($p = 0.5811$) |
| | *(Both automated depth metrics perform $>20\%$ below trivial $81.11\%$ majority baseline; $25.0\%$ scenes unevaluable)* | | |
| **Case Study B (Style Expansion)** | Genuinely Helps CLIP-L | Actively Hurts T5-XXL | Ambiguous / Flat on T5-XXL |
| | *(Consistent with text representation, rather than denoiser architecture, as primary driver of obsolescence)* | | |
| **CFG Rescaling ($\phi = 0.70$)** | Standard Option | Standard Option | Optimal Free Polish ($+0.0398$ LAION) |

---

## 10. Discussion & Limitations

The empirical findings across our two case studies, positive control experiments, and documented failure modes carry direct actionable lessons for both generative benchmark designers and practitioners.

### 10.1 What Benchmark Designers Should Change
1. **Retire Cascaded Detect-then-Evaluate 3D Depth Benchmarks:** Detect-then-depth cascades are vulnerable to detector recall failure on background entities, indicating that benchmarks built on this architecture should report human-calibrated accuracy before their scores are read as measures of generative capability. In our evaluation of an OWL-ViT and Depth Anything V2 cascade, $80.6\%$ of false negatives stemmed from detector dropouts rather than generative misplacements. Benchmark authors must avoid reporting composite scores that conflate detector omissions with spatial positioning failures.
2. **Implement Explicit Scene Abstention:** In $25.0\%$ of generated depth scenes, one or both prompt entities are missing or completely unidentifiable. Automated evaluators assign confident binary verdicts to $100\%$ of these missing-object scenes. Benchmarks must integrate confidence floors (e.g. $\tau \ge 0.12$) or multi-modal abstention mechanisms to flag unevaluable imagery rather than silently scoring invalid generations.
3. **Calibrate Evaluators Against Human Ground Truth:** Automated evaluation pipelines should report baseline accuracy and inter-annotator agreement (such as Cohen's $\kappa$) against blinded human labels before being deployed as standard benchmarks. As shown in Section 5, both evaluated automated depth metrics score $>20\%$ below the trivial majority-class baseline ($81.11\%$), rendering raw metric gains uninterpretable.

### 10.2 What Generative Practitioners Should Stop Doing
1. **Cease Automatic Craft Descriptor Appending:** On modern Diffusion Transformers conditioned on large language models (such as PixArt-$\alpha$ and SD 3.5 Medium), appending photography craft keywords ("cinematic lighting, 35mm, fine grain") is either actively harmful to human preference (ImageReward $\bar{d} = -0.0468$, Wilcoxon $p = 0.0009$; HPS v2.1 $\bar{d} = -0.0015$, Wilcoxon $p = 0.0010$ on PixArt) or fails to reach statistical significance (Wilcoxon $p = 0.1630$ on SD 3.5), while imposing a universal cross-attention semantic dilution cost ($\Delta\text{CLIP} < 0$). Automated prompt expansion wrappers should be disabled by default and preserved strictly as user-controlled stylistic options.
2. **Adopt Variance-Preserving CFG Rescaling as a Default:** High Classifier-Free Guidance scales introduce highlight blowout and oversaturation. Setting CFG rescale to $\phi = 0.70$ provides a modest, verified improvement ($+0.0398$ LAION, $+0.0200$ ImageReward) with zero inference latency overhead.

### 10.3 Methodological Implications for AI-Assisted Research
The four data fabrication incidents documented in Section 8—observed within a single AI coding agent workflow on this project—illustrate that automated research assistants can construct synthetic proxy data and circular verification harnesses when tasked with empirical benchmarking. Because synthetic artifacts mimic realistic score distributions and conform to standard statistical registers, detection cannot rely on numerical plausibility. Computational research workflows utilizing autonomous agents must enforce:
* **Strict Provenance Verification:** Mandatory generation of raw images saved directly to disk.
* **Cryptographic Checksums:** Automated generation of SHA-256 manifests indexing all evaluation artifacts.
* **Independent Execution Harnesses:** Complete separation of evaluation scripts from model code to prevent circular self-auditing.

### 10.4 Limitations
This study is subject to several key constraints:
1. **Hardware Constraints:** All experiments were conducted on a single dedicated consumer GPU (NVIDIA RTX 4060 Ti 16 GB). This constrained batch sizes and precluded full multi-seed evaluation of $8\text{B}+$ parameter models (e.g. SD 3.5 Large).
2. **Single-Annotator Human Ground Truth:** While rigorous and blinded, human validation ground truth in this study rests on a single evaluator ($N = 120$). Although Section 10.1 recommends reporting multi-annotator agreement metrics such as Cohen's $\kappa$ for benchmark deployment, our own dataset does not currently satisfy this standard following the permanent purge of synthetic second-annotator artifacts.
3. **Scope of Architectural Implementations:** Our depth metric audit was conducted specifically on an OWL-ViT and Depth Anything V2 implementation on our 120-image sample; implementations utilizing alternative object detectors (e.g. UniDet, Grounding DINO) on different prompt sets were not evaluated directly.

---

## 11. Conclusion

This study demonstrates that measurement discipline, causal isolation, and provenance verification are vital when evaluating modern generative models. Upgrading from early CLIP encoders to large language models (T5-XXL) eliminates the need for manual prompt engineering tricks while imposing an attention-dilution penalty if they are used. Simultaneously, automated spatial evaluation of 3D depth remains bottlenecked by upstream zero-shot detection failures and scene unidentifiability, even though monocular depth estimators carry informative signal when entities are successfully localized. We hope these negative-heavy, transparently documented findings provide a grounded baseline for future diffusion transformer research.

---

## 12. References

1. **Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., & Cohen-Or, D.** (2023). Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models. *ACM Transactions on Graphics (TOG)*, 42(4), 1–10. [arXiv:2301.13826](https://arxiv.org/abs/2301.13826).
2. **Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., & Li, Z.** (2024). PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. *International Conference on Learning Representations (ICLR)*. [arXiv:2310.00426](https://arxiv.org/abs/2310.00426).
3. **Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., & Rombach, R.** (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. *International Conference on Machine Learning (ICML)*. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206).
4. **Gokhale, T., Palangi, H., Nushi, B., Vineet, V., Horvitz, E., Kamar, E., Baral, C., & Yang, Y.** (2022). Benchmarking Spatial Relationships in Text-to-Image Generation. *arXiv preprint*. [arXiv:2212.10015](https://arxiv.org/abs/2212.10015).
5. **Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., & Cohen-Or, D.** (2022). Prompt-to-Prompt Image Editing with Cross Attention Control. *arXiv preprint*. [arXiv:2208.01626](https://arxiv.org/abs/2208.01626).
6. **Ho, J., & Salimans, T.** (2022). Classifier-Free Diffusion Guidance. *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598).
7. **Huang, K., Sun, K., Xie, E., Li, Z., & Liu, X.** (2023). T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation. *Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023)*. [arXiv:2307.06350v2](https://arxiv.org/abs/2307.06350v2).
8. **Huang, K., Duan, C., Sun, K., Xie, E., Li, Z., & Liu, X.** (2024). T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)* / *arXiv preprint*. [arXiv:2307.06350v3](https://arxiv.org/abs/2307.06350v3).
9. **Lin, S., Liu, B., Li, J., & Yang, X.** (2023). Common Diffusion Noise Schedules and Sample Steps are Flawed. *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2024)*. [arXiv:2305.08891](https://arxiv.org/abs/2305.08891).
10. **Oppenlaender, J.** (2023). A Taxonomy of Prompt Modifiers for Text-To-Image Generation. *Behaviour & Information Technology*, 43(13), 1–16. [arXiv:2204.13988](https://arxiv.org/abs/2204.13988).
11. **Peebles, W., & Xie, S.** (2023). Scalable Diffusion Models with Transformers. *IEEE/CVF International Conference on Computer Vision (ICCV)*, 4195–4205. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).
12. **Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J.** (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research (JMLR)*, 21(140), 1–67. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).
13. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 10684–10695. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752).
14. **Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H.** (2024). Depth Anything V2. *Thirty-eighth Conference on Neural Information Processing Systems (NeurIPS 2024)*. [arXiv:2406.09414](https://arxiv.org/abs/2406.09414).
