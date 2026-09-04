# Hybrid Reasoning-Guided DiT & AI Image Generation Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.30+-orange.svg)](https://github.com/huggingface/diffusers)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A text-to-image and visual editing platform powered by **Diffusion Transformers (PixArt-Alpha DiT)**, **Stable Diffusion**, and a **Hybrid Reasoning & Soft Cross-Attention Spatial Guidance Engine**.

---

## 🔬 Empirical Validation & Status Summary

Live GPU experiments on Stable Diffusion v1.5 (NVIDIA RTX 4060 Ti, CUDA fp16) have established the following empirical boundaries:

1. **Validated Capability — Lateral Spatial Steering ($p = 0.000394$, $N=192$ paired):**
   * 2D cross-attention logit bias produces **statistically significant control over lateral (left/right/beside) object placement** at strength 6.0, improving satisfaction from $34.90\% \to 49.48\%$ ($+28$ net paired gains across 24 prompts $\times$ 8 seeds, McNemar exact $p = 3.94 \times 10^{-4}$).
   * *Status:* **Validated & Promoted to Default.** Human visual review confirmed image quality at strength 6.0 is clean (no duplication, distortion, or texture breakdown; low SSIM reflects compositional reorganization rather than damage, with ON fixing baseline artifacts in several pairs). Object-presence audits across all 192 pairs show dual entity presence increases ($59.38\% \to 67.71\%$) and omissions drop ($40.6\% \to 32.3\%$). Manual ground-truth labeling on 30 ON images confirmed 100% detector precision and an underlying true success rate of $63.33\%$.
2. **Negative Result — 3D Camera Depth Control ($p = 0.081$, $N=192$ paired):**
   * When measured with a true 3D monocular depth estimator (**Depth Anything V2**), depth guidance does **not** achieve statistical significance ($41.67\% \to 47.92\%$, McNemar exact $p = 0.0807$).
   * The apparent gain on the 2D bounding-box metric ($50.0\% \to 60.4\%$, $p = 0.0029$) is an artifact of objects moving lower in the image frame—a 2D pictorial depth cue—rather than true camera-space distance modulation.
   * *Status:* **Disabled by default (`DEPTH_RELATION_GUIDANCE_STRENGTH = 0.0`).**
3. **Negative Result — Vertical-On Placement ($p = 0.453$):**
   * Unguided SD v1.5 already possesses a strong natural resting prior ($70.83\%$). Imposing cross-attention bias trends negative ($70.83\% \to 58.33\%$, $b=2, c=5$ at 6.00).
   * *Status:* **Disabled by default (`VERTICAL_ON_GUIDANCE_STRENGTH = 0.0`).**
4. **Depth Subsystem Architecture Status:**
   * Features including $\mu_z$ depth coordinates, relative depth DAG solving, continuous density fields, 3D Gaussian priors, and depth-aware overlap modulation are fully implemented and unit-tested in code, but have **no demonstrated effect on real generated 3D camera depth**. They remain in the codebase marked as **unvalidated**.

Full experimental datasets, paired contingency tables, and visual review artifacts are documented in [docs/experiments.md](docs/experiments.md).

---

## 🌟 Key Highlights & Core Capabilities

### 1. Lateral Cross-Attention Spatial Guidance (Validated)
- **2D Coordinate Control:** Directs entity tokens toward planned horizontal bounding regions using additive cross-attention bias during early denoising phases ($0\% - 80\%$).
- **Statistically Significant Improvement:** Confirmed on powered paired tests ($N=192$, $p < 0.001$).
- **Aesthetic Isolation:** Style, lighting, and medium tokens strictly receive $0.0$ bias to preserve diffusion texture.

### 2. Experimental / Unvalidated Depth & Density Subsystems
- **Volumetric Spatial Priors (Unvalidated on real 3D depth):** Parameterizes entities with centroid $(\mu_y, \mu_x, \mu_z)$ and anisotropic scale in $[0.0, 1.0]^3$.
- **Continuous Density Field Modeling (Unvalidated on real swarms):** Models high-count ensembles ($\ge 10$ instances) via differentiable continuous distributions (Gaussian, Uniform Plateau, Radial, Streamline).
- **Direct Spatial Visual Feature Cross-Attention:** Adapter architecture injecting localized reference visual features directly into cross-attention keys/values.

### 3. Interactive Web Canvas
- **Direct Manipulation:** Built-in HTML5 canvas supporting real-time drag-and-drop, 8-point corner resizing, 360° interactive rotation handles ($\theta \in [-\pi, \pi]$), and relative depth ($z$) adjustment.

### 4. Multi-Pass Conversational Image Assistant
- **Two-Pass Analytical Decoupling:** Decouples fast analytical reasoning (Pass 1: intent, ambiguity resolution) from user-facing conversational response generation (Pass 2), eliminating prompt leakage.

### 5. Tri-Tier Region-Aware Image Editing
- **Anti-Leakage Architecture:** Combines token role alignment, spatial classifier-free guidance, and scheduled latent blending for localized image modifications.

---

## 🏛️ System Architecture

```
                    Prompt (+ Optional Reference Image / Interactive Canvas Overrides)
                                                    │
                                                    ▼
       ┌────────────────────────────────────────────────────────────────────────────────────────┐
       │                STAGE 1: Spatial & Relation Semantic Planner                            │
       │                                (semantic_planner.py)                                   │
       ├────────────────────────────────────────────────────────────────────────────────────────┤
       │ • Relation Parsing: Classifies lateral, depth, vertical_on, and vertical_under         │
       │ • Per-Relation Strength Dispatch: lateral=6.0, depth=0.0, vertical_on=0.0, under=0.3   │
       │ • Unvalidated Subsystems: 3D Gaussian priors (mu_z), relative depth DAG, density fields│
       │ • Aesthetic Tokens Isolation: Style, lighting, and mood strictly receive 0.0 bias       │
       └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    │
                             Structured Plan (Objects, Densities, Overlaps)
                                                    │
                                                    ▼
       ┌────────────────────────────────────────────────────────────────────────────────────────┐
       │               STAGE 2: Cross-Attention Spatial Guidance Processor                      │
       │                           (layout_guidance.py, vision_backbone.py)                     │
       ├────────────────────────────────────────────────────────────────────────────────────────┤
       │ • Per-Relation Guidance Bias: Applies +gamma[rel] * Heatmap[obj]                       │
       │ • Lateral Path (Active): +6.0 bias steers left/right coordinates (p = 0.000394)        │
       │ • Depth & Vertical-On Paths (Bypassed by default): 0.0 bias (unconstrained prior)      │
       │ • Dynamic Schedule: TwoPhaseSchedule cuts off guidance at t >= 0.80 for fine detailing │
       └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    │
                          Drop-in Attention Hook (Training-Free, Diffusers-Native)
                                                    │
                                                    ▼
       ┌────────────────────────────────────────────────────────────────────────────────────────┐
       │                    STAGE 3: Diffusion Transformer (DiT) Denoising Loop                 │
       │                                   (edit_pipeline.py)                                   │
       ├────────────────────────────────────────────────────────────────────────────────────────┤
       │ • Denoising steps 0..T: Soft guidance anchors horizontal positions                     │
       │ • DiT transformer blocks synthesize photorealistic textures and artistic lighting      │
       └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                    │
                                                    ▼
                                       VAE Decode -> Final Image
```

---

## 📊 Offline Mathematical & Structural Invariant Benchmarks

> **Note on Benchmark Methodology:** The metrics below are evaluated via the offline structural invariant test harness (`scripts/eval_hybrid_reasoning.py` and `scripts/eval_editing.py`). They measure **mathematical and algebraic invariants** (attention bias matrix isolation, 2D/3D coordinate calculations, and latent blending equations on synthetic 2D latent fields with a toy 12-step Euler loop). They demonstrate theoretical bounds and software invariants; **they are not perceptual evaluations of generated images**. Live empirical image benchmarks are documented in [docs/experiments.md](docs/experiments.md).

| Failure Mode / Benchmark Category | Evaluated Conditions | Baseline Model | Proposed Hybrid Framework | Offline Verification Type |
|---|---|---|---|---|
| **1. Object Count Accuracy** | Single, multi-words, digit numerals, mixed quantifiers, collective nouns, numeral sequences | Common count confusion & duplicate blending | **100.0%** (15/15 entities exact match) | **Deterministic Rule/Parser Invariant** |
| **2. Spatial Relation Correctness** | `riding` (fwd/rev), `under`, `next_to`, `inside`, `in_front_of`, `behind`, unlinked partition | Spatial inversions and semantic bleeding | **100.0%** (8/8 spatial geometries correct) | **Geometric Coordinate Verification** |
| **3. Next-Gen Spatial & Depth Reasoning** | 3D Gaussians, relative depth, continuous swarms, star fields, 45° rotations, visual features | Hard box bounds / no depth reasoning | **100.0%** (12/12 complex next-gen cases) | **Continuous Potential Field Invariant** |
| **4. Edit Target Isolation & Anti-Leakage** | Local recoloring, small objects, background preservation, regional sky changes | Toy Leakage: `0.561`<br>Toy SSIM Out: `0.832`<br>IoU: `0.439` | Toy Leakage: **`0.006`**<br>Toy SSIM Out: **`0.998`**<br>IoU: **`0.889`** | **Synthetic Latent Blend Invariant** |
| **5. Aesthetic Control Set (Zero Bias)** | Cyberpunk watercolor, cinematic portrait, macro photorealism, whimsical anime, oil sunset, pixel art | Uncontrolled style pinning and texture loss | **100.0%** (Zero spatial bias verified on all style tokens) | **Attention Logit Masking Invariant** |
| **6. Guidance Ablation & Entropy Retention** | Soft guidance ($+0.3$) vs hard masking ($-12.0$) | Hard Gradient Ret: **`0.00%`** (Total Collapse) | Soft Gradient Ret: **`100.0%`**<br>Soft Entropy Ret: **`100.0%`** | **Softmax Analytical Gradient Invariant** |

---

## 🚀 Quickstart & Installation

### Requirements
- **Python:** 3.10 or higher
- **GPU:** NVIDIA GPU with 16 GB+ VRAM recommended for PixArt-Alpha DiT inference (CPU mode fully supported for API development, testing, and Stable Diffusion).

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/quang8112009/midjourney.git
cd midjourney

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

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

The frontend web UI (`frontend/index.html`) includes a full interactive layout studio:

1. **Auto-Plan from Prompt:** Click `⚡ Auto-Plan from Prompt` to parse nouns, counts, and spatial relationships into 3D Gaussian priors.
2. **Interactive Rotation:** Click and drag the circular rotation handle above any entity to rotate its anisotropic Gaussian prior ($\theta$).
3. **Depth Controls:** Configure relative depth ($\mu_z \in [0.0, 1.0]$) to adjust foreground and background hierarchy.
4. **Gaussian Heatmap Preview:** Real-time radial gradient rendering showing soft cross-attention bias.

---

## 🔌 API Reference & Usage

### 1. Image Generation (`POST /api/v1/generate`)

```json
{
  "model": "pixart-alpha",
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
  "guidance_scale": 4.5
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

The test suite runs completely offline without requiring heavy model downloads:

```bash
# Run all unit and integration tests (350+ tests)
python -m unittest discover -s tests

# Run code style & linting check
ruff check .

# Run the complete failure mode benchmark suite
python scripts/eval_hybrid_reasoning.py
```

---

## ⚙️ Configuration Reference (`.env`)

```dotenv
# Core Model Settings
MODEL_ID=runwayml/stable-diffusion-v1-5
PIXART_MODEL_ID=PixArt-alpha/PixArt-XL-2-512x512
MODEL_CACHE_DIR=./models/cache
OUTPUT_DIR=./outputs
DEVICE=auto
DTYPE=auto
MODEL_CPU_OFFLOAD=true

# Spatial, Depth & Multi-Modal Guidance
DEPTH_GUIDANCE_ENABLED=true
DEPTH_GUIDANCE_STRENGTH=0.3  # Legacy fallback / global default
LATERAL_GUIDANCE_STRENGTH=6.0  # Validated lateral steering default
DEPTH_RELATION_GUIDANCE_STRENGTH=0.0  # Disabled (unvalidated on real 3D depth)
VERTICAL_ON_GUIDANCE_STRENGTH=0.0  # Disabled (unguided prior is stronger)
VERTICAL_UNDER_GUIDANCE_STRENGTH=0.3  # Preserved default
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

## 🐳 Docker Deployment

```bash
# Build and run container with GPU acceleration
docker compose up --build
```

---

## 📚 Technical Documentation

- [docs/soft-guidance-tuning.md](docs/soft-guidance-tuning.md): 3D Gaussian math, continuous density fields, guidance schedules, and visual feature projection.
- [docs/visual-reasoning-editing.md](docs/visual-reasoning-editing.md): Tri-tier region-aware image editing and leakage prevention.
- [docs/two-pass-reasoning.md](docs/two-pass-reasoning.md): Conversational two-pass analytical reasoning pipeline.
- [docs/dit-research.md](docs/dit-research.md): Diffusion Transformer architecture review and training-free attention hooks.
- [docs/data-flow.md](docs/data-flow.md): End-to-end data flow specification.

---

## 📄 License

Distributed under the [MIT License](LICENSE).
