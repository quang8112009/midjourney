# Hybrid Reasoning-Guided DiT & AI Image Generation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.30+-orange.svg)](https://github.com/huggingface/diffusers)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance text-to-image and visual editing platform powered by **Multimodal Diffusion Transformers (MMDiT / Stable Diffusion 3.5)**, **PixArt-Alpha DiT**, **Stable Diffusion v1.5**, and a **Hybrid Reasoning & Soft Attention Spatial Guidance Engine**.

---

## 🔬 Empirical Validation & Research Status

Live GPU experiments on NVIDIA GeForce RTX 4060 Ti (16GB VRAM, CUDA fp16) have established the following empirical boundaries across model backbones:

### 1. Diffusion Transformer Upgrade: SD 3.5 Medium (MMDiT Architecture)
- **Model Backbone:** `stabilityai/stable-diffusion-3.5-medium` (2.5B parameter MMDiT transformer, 24 Joint Transformer Blocks).
- **VRAM & Compute Efficiency:**
  - Full model FP16 footprint is **14.73 GB** (`transformer`: 4.18 GB, `text_encoder_3` / T5-XXL: 8.87 GB, `text_encoder_2` / CLIP-G: 1.29 GB, `text_encoder` / CLIP-L: 0.23 GB, `vae`: 0.16 GB).
  - Peak allocated VRAM during inference is **12.07 GB** (Reserved: **12.20 GB**), fitting natively within 16GB consumer VRAM with zero PCIe memory thrashing.
  - Generates at **0.195 s/step** ($5.11\text{ it/s}$, $\sim 3.9\text{s}$ per image denoising) at $512\times 512$ / 20 steps, and **0.800 s/step** ($1.25\text{ it/s}$, $\sim 22.4\text{s}$ per image denoising) at $1024\times 1024$ / 28 steps.
- **MMDiT Joint Attention Hooking:**
  - Hooks attach cleanly across all **37 attention processors** (Blocks 0–12 dual-stream `attn` + `attn2`, Blocks 13–23 joint-stream `attn`).
  - Implements scaled dot-product attention bias targeting the off-diagonal $Q_{\text{img}} K_{\text{txt}}^T$ slice without attention leakage.
- **Multi-Encoder Token Mapping & Aesthetic Isolation:**
  - Maps joint context across 666 total tokens: **CLIP-L** (`[0..76]`, 77 tokens), **CLIP-G** (`[77..153]`, 77 tokens), and **T5-XXL** (`[154..665]`, 512 tokens).
  - **Aesthetic Token Isolation:** Empirically verified on real GPU tensors: **`0.0000000` max bias** on CLIP-L style and CLIP-G lighting tokens, ensuring full preservation of diffusion texture and aesthetic priors while spatial guidance operates strictly on T5-XXL entity tokens.

### 2. SentencePiece Tokenizer Regression Resolution
- **Root Cause:** In SentencePiece tokenization (T5-XXL), standalone whitespace prefix tokens (`\u2581` / `_`) before prepositions/articles caused word-index state machine drift, resulting in 31.2% (29/93) of planned entities in the 24 lateral benchmark prompts dropping to empty token lists `()`.
- **Resolution:** Re-engineered `map_pieces_to_words` with prefix-marker tracking and implemented `_resolve_label_token_indices()` for compound noun phrases.
- **Result:** **100% entity resolution rate (93/93 planned entities)** across CLIP-L, CLIP-G, and T5-XXL, validated via comprehensive unit tests in `tests/test_dit_enhancements.py`.

### 3. MMDiT Powered Spatial Guidance Benchmark (SD 3.5 Medium, $N=192$ Paired Runs)

Cross-architecture transfer of soft spatial cross-attention guidance was established across two $N=192$ benchmark suites ($512\times 512$ / 20 Euler steps, 8 seeds):

#### Benchmark Suite Results ($N=192$ Paired Runs per Condition)

| Benchmark Suite | Condition | Overall Satisfaction | Wilson 95% CI | Directional Rate | Dual Presence | Misplaced / Omitted | McNemar $p$-value vs OFF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard 24** | **OFF (0.00)** | $83.85\%$ ($161/192$) | $[78.0\%, 88.4\%]$ | $80.88\%$ ($110/136$) | $94.79\%$ | $21$ / $10$ | — (Baseline Ref) |
| **Standard 24** | **ON (3.00)** | **$91.15\%$** ($175/192$) | $[86.3\%, 94.4\%]$ | **$90.44\%$** ($123/136$) | **$96.35\%$** | **$10$** / **$7$** | **$p = 0.002577$ (Signif.)** |
| **Standard 24** | **ON (6.00)** | $89.58\%$ ($172/192$) | $[84.5\%, 93.2\%]$ | $87.50\%$ ($119/136$) | $94.27\%$ | $9$ / $11$ | $p = 0.098872$ |
| **Hard 24** | **OFF (0.00)** | $52.08\%$ ($100/192$) | $[45.1\%, 59.0\%]$ | $52.08\%$ ($100/192$) | $77.60\%$ | $49$ / $43$ | — (Baseline Ref) |
| **Hard 24** | **ON (3.00)** | $59.38\%$ ($114/192$) | $[52.3\%, 66.1\%]$ | $59.38\%$ ($114/192$) | **$80.21\%$** | $40$ / $38$ | **$p = 0.028817$ (Signif.)** |
| **Hard 24** | **ON (6.00)** | **$61.46\%$** ($118/192$) | $[54.4\%, 68.1\%]$ | **$61.46\%$** ($118/192$) | **$80.21\%$** | **$36$** / $38$ | **$p = 0.022241$ (Signif.)** |

#### Backbone Evolution: Unaided Directional Spatial Baseline
* **SD v1.5 Directional Baseline (OFF, $N=136$):** $27.94\%$ ($38/136$)
* **SD 3.5 Medium Directional Baseline (OFF, $N=136$):** **$80.88\%$** ($110/136$)
* **Unaided Evolution:** $+52.94\%$ absolute improvement from modern multimodal text encoding (T5-XXL) and 2.5B MMDiT transformer capacity.

#### Operating Strength Recommendation
* **Statistical Separability:** Wilson 95% confidence intervals for strength 3.0 and 6.0 overlap heavily across both suites ($[86.3\%, 94.4\%]$ vs $[84.5\%, 93.2\%]$ on Standard; $[52.3\%, 66.1\%]$ vs $[54.4\%, 68.1\%]$ on Hard). Neither strength is statistically separable as a universal default at this sample size.
* **Task Distribution Selection:**
  - **Strength 3.00:** Well-suited for standard scenes where the base model already exhibits strong spatial comprehension ($80.88\% \to 90.44\%$, $p=0.00258$), minimizing over-steering.
  - **Strength 6.00:** Provides stronger spatial steering on complex, cluttered, or counter-prior compositions.

### 4. Established UNet Spatial Boundaries (Stable Diffusion v1.5)
- **Lateral Spatial Steering ($p = 0.000394$, $N=192$ paired):** Statistically significant horizontal control ($34.90\% \to 49.48\%$, $+28$ net paired gains across 24 prompts $\times$ 8 seeds, McNemar $p = 3.94 \times 10^{-4}$).
- **3D Camera Depth Control ($p = 0.081$, $N=192$ paired):** Evaluated with **Depth Anything V2**, depth guidance did not achieve statistical significance ($41.67\% \to 47.92\%$, $p = 0.0807$). Disabled by default (`DEPTH_RELATION_GUIDANCE_STRENGTH = 0.0`).
- **Vertical-On Placement ($p = 0.453$):** Unguided model already exhibits a strong resting prior ($70.83\%$). Disabled by default (`VERTICAL_ON_GUIDANCE_STRENGTH = 0.0`).

Full datasets, paired contingency tables, forensic sub-group breakdowns, and visual review artifacts are documented in [docs/experiments.md](docs/experiments.md) and [docs/dit-research.md](docs/dit-research.md).

---

## 🌟 Architecture & Key Capabilities

```
                    Prompt (+ Optional Interactive Canvas Overrides / Reference Image)
                                                     │
                                                     ▼
        ┌────────────────────────────────────────────────────────────────────────────────────────┐
        │                STAGE 1: Spatial & Relation Semantic Planner                            │
        │                                (semantic_planner.py)                                   │
        ├────────────────────────────────────────────────────────────────────────────────────────┤
        │ • Relation Parsing: Classifies lateral, depth, vertical_on, and vertical_under         │
        │ • Token Resolution: SentencePiece & BPE-aware mapping with compound noun support       │
        │ • Multi-Encoder Token Isolator: Enforces strict 0.0 bias on CLIP-L (77) and CLIP-G (77)│
        │ • Target Extraction: Directs spatial priors exclusively to T5-XXL entity tokens (512)   │
        └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                     │
                              Structured Plan (Objects, Densities, Overlaps)
                                                     │
                                                     ▼
        ┌────────────────────────────────────────────────────────────────────────────────────────┐
        │               STAGE 2: MMDiT & UNet Attention Guidance Processors                      │
        │                           (layout_guidance.py, vision_backbone.py)                     │
        ├────────────────────────────────────────────────────────────────────────────────────────┤
        │ • MMDiT Joint Attention Hook: Modifies Q_img K_txt^T off-diagonal slice (37 modules)   │
        │ • Per-Relation Guidance Strength: Lateral steering dispatched dynamically              │
        │ • Dynamic Schedule: TwoPhaseSchedule cuts off guidance at t >= 0.80 for fine texture   │
        └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                     │
                           Drop-in Attention Hook (Training-Free, Diffusers-Native)
                                                     │
                                                     ▼
        ┌────────────────────────────────────────────────────────────────────────────────────────┐
        │                 STAGE 3: Diffusion Transformer (DiT / MMDiT) Denoising Loop            │
        │                                   (edit_pipeline.py)                                   │
        ├────────────────────────────────────────────────────────────────────────────────────────┤
        │ • FlowMatchEuler / DPMSolverMultistep sampling across scheduled reverse-time steps     │
        │ • DiT transformer blocks synthesize photorealistic textures and artistic lighting      │
        └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                     │
                                                     ▼
                                        VAE Decode -> Final Image
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python:** 3.10 or 3.11
- **GPU:** NVIDIA GPU with 16 GB+ VRAM recommended for native FP16 MMDiT / DiT inference. (CPU mode supported for API testing and development).

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/quang8112009/midjourney.git
cd midjourney

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Copy environment configuration
cp .env.example .env
```

### 2. Start the Application Server

```bash
python main.py
```

- **Interactive Web UI:** [http://localhost:8000/ui/](http://localhost:8000/ui/)
- **Interactive OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🎨 Interactive Layout Editor

The frontend web UI (`frontend/index.html`) includes a layout canvas:
1. **Auto-Plan from Prompt:** Parses nouns, counts, and spatial relationships into 2D/3D Gaussian priors.
2. **Interactive Rotation:** Drag-and-drop rotation handles to rotate anisotropic Gaussian priors ($\theta$).
3. **Depth Controls:** Configure relative depth ($\mu_z \in [0.0, 1.0]$) for foreground/background hierarchy.
4. **Gaussian Heatmap Preview:** Real-time visual feedback rendering soft cross-attention bias.

---

## 🔌 API Reference & Usage

### 1. Image Generation (`POST /api/v1/generate`)

```json
{
  "model": "stable-diffusion-3.5",
  "prompt": "a red ball in front of a blue cube, cinematic lighting, 8k",
  "guidance_mode": "gaussian",
  "adaptive_guidance": true,
  "layout_override": [
    {
      "label": "ball",
      "count": 1,
      "ymin": 0.35, "xmin": 0.20, "ymax": 0.90, "xmax": 0.80,
      "theta": 0.0,
      "mu_z": 0.25,
      "entity_id": "ball_01"
    },
    {
      "label": "cube",
      "count": 1,
      "ymin": 0.10, "xmin": 0.25, "ymax": 0.60, "xmax": 0.75,
      "theta": 0.7854,
      "mu_z": 0.70,
      "entity_id": "cube_01"
    }
  ],
  "num_inference_steps": 20,
  "guidance_scale": 4.5,
  "width": 512,
  "height": 512
}
```

### 2. Pre-Denoise Layout Planning (`POST /api/v1/layout/plan`)

```http
POST /api/v1/layout/plan
Content-Type: application/json

{
  "prompt": "three red apples and two green pears on a rustic wooden table",
  "guidance_mode": "gaussian",
  "adaptive_guidance": true
}
```

### 3. Conversational Assistant (`POST /api/v1/chat`)

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "Generate a cozy coffee shop in the rain, watercolor style",
  "session_id": "user_session_123"
}
```

---

## 🧪 Testing & Verification

Run the comprehensive unit and integration test suite:

```bash
# Run full pytest test suite
pytest tests/

# Run DiT enhancement and tokenizer invariant tests
pytest tests/test_dit_enhancements.py

# Run code style & linting check
ruff check .
```

---

## ⚙️ Configuration Reference (`.env`)

```dotenv
# Core Model Settings
MODEL_ID=runwayml/stable-diffusion-v1-5
PIXART_MODEL_ID=PixArt-alpha/PixArt-XL-2-512x512
SD35_MODEL_ID=stabilityai/stable-diffusion-3.5-medium
MODEL_CACHE_DIR=./models/cache
OUTPUT_DIR=./outputs
DEVICE=auto
DTYPE=auto
MODEL_CPU_OFFLOAD=true

# Spatial & Multi-Modal Guidance
DEPTH_GUIDANCE_ENABLED=true
DEPTH_GUIDANCE_STRENGTH=0.3
LATERAL_GUIDANCE_STRENGTH=6.0
DEPTH_RELATION_GUIDANCE_STRENGTH=0.0
VERTICAL_ON_GUIDANCE_STRENGTH=0.0
VERTICAL_UNDER_GUIDANCE_STRENGTH=0.3
SELF_ATTENTION_DEPTH_BIAS_ENABLED=true
DENSITY_FIELD_ENABLED=true
DENSITY_ENTITY_THRESHOLD=10
VISUAL_CROSS_ATTN_ENABLED=true
VISUAL_FEATURE_STRENGTH=0.25
VISION_BACKBONE=auto
ROTATION_EDITING_ENABLED=true

# Rate Limits & Budgets
MAX_BATCH_PIXELS=1048576
MAX_GENERATION_WORK_UNITS=52428800
RATE_LIMIT_PER_MINUTE=10

# Conversational Assistant
CHAT_PROVIDER_TYPE=openai
CHAT_API_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini
CHAT_TWO_PASS_ENABLED=true
```

---

## 📚 Technical Documentation

- [docs/dit-research.md](docs/dit-research.md): Diffusion Transformer research, MMDiT joint attention hooks, and multi-backbone comparison.
- [docs/soft-guidance-tuning.md](docs/soft-guidance-tuning.md): 3D Gaussian math, continuous density fields, and guidance schedules.
- [docs/experiments.md](docs/experiments.md): Comprehensive empirical benchmark results, contingency tables, and detector audits.
- [docs/visual-reasoning-editing.md](docs/visual-reasoning-editing.md): Tri-tier region-aware image editing and leakage prevention.
- [docs/two-pass-reasoning.md](docs/two-pass-reasoning.md): Conversational two-pass analytical reasoning pipeline.
- [docs/data-flow.md](docs/data-flow.md): End-to-end system data flow specification.

---

## 📄 License

Distributed under the [MIT License](LICENSE).
