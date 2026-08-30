# Region-aware visual reasoning for DiT editing

## 1. Analysis of the current pipeline

The goal template assumed an existing editing pipeline with a reference/conditioning
mechanism to tune. This repository does not have one. What exists today
(`app/services/model_service.py`) is **text-to-image only**:

| Assumed by the goal | Actually present |
|---|---|
| img2img / inpainting path | none - `generate_image()` takes a prompt, never an image |
| reference image conditioning (VAE concat / IP-Adapter / ControlNet) | none |
| tunable reference coefficient | a single scalar `guidance_scale` (7.5 SD, 4.5 PixArt) |
| cross-attention hooks | none - the pipeline is called as a black box |
| DiT | `PIXART_ALPHA` (`PixArt-XL-2-512x512`) is the DiT; `STABLE_DIFFUSION` is a U-Net |

So the four stated problems could not be "fixed" in existing code - the code where they
would occur does not exist yet. The work below **builds the editing path** with the
requested mechanisms designed in from the start, at the conditioning level only, so the
base DiT is untouched and an existing checkpoint still loads.

The underlying failure being designed against is real and structural: in a DiT every text
token attends to every image token, so nothing in "change the shirt color to red" tells the
model *where* the shirt is. A single scalar guidance applies that prompt uniformly to the
whole frame - which is exactly the leakage described.

## 2. Architecture

Three interventions, all training-free, all outside the transformer weights (see `docs/soft-guidance-tuning.md` for soft layout guidance theory, two-phase scheduling, and parameter tuning):

```
                     source image + prompt (+ optional user mask, scene facts)
                                        |
                          +-------------v--------------+
                          |  plan_edit()  - ONE pass   |   edit_planner.py
                          |  region | scope | strengths|
                          +-------------+--------------+
                                        |
                     +------------------+------------------+
                     |                                     |
          +----------v-----------+              +----------v-----------+
          | alignment check      |  conflict -> | clarify / assume     |  alignment.py
          | (pre-denoise)        |              | 0 denoise steps spent|
          +----------+-----------+              +----------------------+
                     | ok
                     v
        +============================================+
        |            denoise loop (per step)         |   edit_pipeline.py
        |                                            |
        |  1. region-aware cross-attention bias      |   region_attention.py
        |     edit tokens suppressed outside region  |
        |                                            |
        |  2. spatial CFG                            |   adaptive_reference.py
        |     noise = uncond + scale_map*(cond-uncond)|
        |     scale_map: inside > base > outside     |
        |                                            |
        |  3. scheduled latent blending              |
        |     outside restored toward source,        |
        |     ramping in after the layout steps      |
        +============================================+
                     |
                     v                VAE decode -> image
```

**Why three and not one.** They fail differently, so they cover each other: attention
masking stops the *cause* but is soft; spatial CFG steers the *amount*; latent blending is
the hard structural guarantee that pixels outside the region come back. Ablation is
supported (`blend=False`) and tested.

## 3. Modules

| Module | Responsibility |
|---|---|
| `masks.py` | soft masks, feather/dilate, latent-grid resize, IoU, bounding box |
| `region_attention.py` | role tagging, `extract_edit_mask`, role-aware bias, `masked_cross_attention` |
| `edit_planner.py` (tokens) | `map_pieces_to_words` segments sub-words; roles/indices share it |
| `adaptive_reference.py` | locality/conflict -> coefficients, spatial guidance map, edge blending |
| `alignment.py` | pre-denoise prompt/image conflict and ambiguity check |
| `edit_planner.py` | the single planning pass that produces an `EditPlan` |
| `edit_pipeline.py` | the region-aware denoise loop, plus the baseline loop for comparison |
| `metrics.py` | leakage, region IoU, SSIM/L1 outside the mask, in-region alignment |

### Token roles and the shared mask

`classify_token_roles` tags each prompt word `edit_target` / `context` / `neutral` with a
heuristic: words after an edit verb are targets, words after a preservation cue
("...**but keep the** background neutral") are context. `build_attention_bias` then
penalises targets *outside* the region and boosts context *inside* it.

```
"change the jacket to red but keep the background neutral"
 neutral neutral  TARGET  n  TARGET  n   n   n     CONTEXT   CONTEXT
```

Both the attention bias and the guidance map are built from **`plan.mask`** - the single
mask `plan_edit` resolved - so the two mechanisms can never disagree about where the region
is. A test asserts this directly (`SharedMaskTests`).

A learned or POS-based classifier can replace the heuristic without touching the bias
builder; the interface is the role list.

### `leak_penalty` is -12, not -1e4

The reference sketch uses `-1e4`. Two problems, both measured: it saturates the softmax at
*any* strength, so the knob cannot be tuned at all (0.25 and 1.0 give identical output); and
in fp16 it overflows to `-inf`, producing NaNs when a row is fully masked. -12 nats is
~6e-6 relative weight - effectively a hard mask - while staying finite and graded. A test
pins the fp16 finiteness.

### The reference pseudo-code has a sign error

The supplied design computes inside guidance as `base * (1 - ref_weight) * 2`. Since
`ref_weight` rises as the edit gets *more local*, a small local edit - the case that needs a
decisive local change - receives the **weakest** prompt guidance. Measured:

| edit | legacy inside scale | this implementation |
|---|---|---|
| small local, high conflict | 5.32 (below the 7.5 base) | 11.70 |
| global, high conflict | 11.11 | 11.70 |

Also note `(1 - conflict_score)` is algebraically just `similarity`, so the second term was
doing less than it appears.

The fix separates the two jobs the single coefficient was doing:

* `ref_weight` (from **locality**) governs *outside*: preservation and blend strength.
* `edit_strength` (from **conflict**) governs *inside*: how hard to follow the prompt.

The original formula is kept as `legacy_reference_coefficient` so the eval can price it, and
a test pins the difference so the fix cannot be silently reverted.

### Similarity calibration

Text-image cosine similarity occupies a narrow band (CLIP is typically ~0.15-0.35), so the
raw value barely moves any weighted sum. `calibrate_similarity` maps
`[similarity_floor, similarity_ceiling]` onto `[0, 1]`. **Recalibrate these two numbers for
your encoder** - they are the most deployment-specific knob here.

## 4. Before/after results

`python scripts/eval_editing.py` - CPU, offline, no checkpoint, 5 seeds per case.

Both arms call an **identical simulated denoiser** that applies the prompt direction to the
whole frame (the modelled failure). Only the conditioning differs, so the delta is
attributable to the mechanism.

The simulator is **attention-coupled**: an edit lands at a position in proportion to the
share of cross-attention its `edit_target` tokens receive there. With no bias that share is
uniform, so the edit lands everywhere - the leak. This is what makes `leak_penalty` and
`context_boost` tunable against measured outcomes instead of merely asserted.

| arm | leakage ↓ | region IoU ↑ | SSIM outside ↑ | L1 outside ↓ | edit inside |
|---|---|---|---|---|---|
| baseline (scalar guidance, today) | 0.561 | 0.438 | 0.823 | 0.117 | 0.131 |
| legacy (supplied pseudo-code) | 0.525 | 0.438 | 0.894 | 0.068 | 0.143 |
| **proposed** | **0.006** | **0.889** | **0.998** | **0.0001** | 0.158 |

Per case, the local edits are where the mechanism earns its keep:

| case | baseline leakage | proposed leakage | baseline IoU | proposed IoU |
|---|---|---|---|---|
| local: shirt recolor (6.3% of frame) | 0.921 | 0.005 | 0.078 | 0.906 |
| local: small object (2.4%) | 0.965 | 0.008 | 0.034 | 0.916 |
| local+context: preserve background (7.9%) | 0.902 | 0.007 | 0.097 | 0.818 |
| regional: sky (40%) | 0.578 | 0.003 | 0.422 | 0.804 |
| global: watercolor restyle | 0.000 | 0.000 | 1.000 | 1.000 |
| conflicting: 2nd person, 3 present | blocked before denoising - 0 steps spent | | | |

Reading these:

* **Leakage down 99%** overall, and the edit *inside* the region got **stronger**
  (0.131 -> 0.158), not weaker - it is not buying cleanliness by refusing to edit.
* **Global edits are untouched** (identical across arms). Correct: `attention_strength` is 0
  and the full-frame mask makes blending a no-op, so restyling is not sabotaged.
* **Legacy sits between the two** and *under-edits* the local+context case (0.043 inside vs
  the 0.055 baseline) - the sign error, visible in the numbers.
* The conflicting prompt never reaches the denoiser.

### Tuning results

`python scripts/eval_editing.py --sweep` (local and regional cases only - global cases
cannot leak, so they carry no tuning signal).

`leak_penalty` behaves as a clean trade-off: more penalty removes more leakage but slowly
weakens the in-region edit, because a saturated row leaves the edit tokens less mass
everywhere.

| leak_penalty | leakage ↓ | edit inside ↑ |
|---|---|---|
| -2 | 0.138 | 0.138 |
| -4 | 0.043 | 0.134 |
| -8 | 0.013 | 0.130 |
| -12 (default) | 0.009 | 0.127 |
| -20 | 0.006 | 0.122 |

Requiring the edit to stay within 5% of the strongest observed, the sweep picks
**`leak_penalty = -4`**. The shipped default of -12 buys another 5x leakage reduction for
about 5% of edit strength; pick along that frontier for your content.

**`context_boost` measures as a cost, not a benefit** - every increase slightly *raises*
leakage and *lowers* the in-region edit:

| context_boost | leakage ↓ | edit inside ↑ |
|---|---|---|
| 0.0 | 0.0086 | 0.1265 |
| 0.5 (sketch default) | 0.0086 | 0.1248 |
| 1.5 | 0.0087 | 0.1204 |

This is mechanical: softmax is zero-sum, so boosting context tokens inside the region takes
mass from the edit tokens there. **The harness cannot see the upside** - it models a single
scalar edit direction and has no separate "context" semantics for the boost to protect. So
this is *not* evidence that `context_boost` is harmful, only that its benefit is
unmeasurable here. The default stays at the sketch's 0.5 pending a real-model test; if you
must choose on evidence available today, 0.0 is what the numbers support.

The locality/similarity split is nearly flat (leakage 0.0091 -> 0.0082 across 0.4/0.6 to
0.8/0.2), because leakage is dominated by the attention bias rather than by the coefficient.
The 0.6/0.4 default is fine; this knob matters more for preservation than for leakage.

## 5. Limitations and next steps

Honest accounting of what these numbers do and do not establish:

1. **The eval is a simulator, not a model run.** It validates the *mechanism* under a
   modelled failure. It cannot predict perceptual quality. Real numbers need a GPU with
   `PixArt-XL-2-512x512`; this box is CPU-only and the service already gates PixArt behind
   CUDA. Nothing here has been run against a real checkpoint.
2. **In-region alignment saturates at 1.000** because the simulator applies a pure direction
   with no distortion. On a real model this is the metric most likely to move, and it is the
   one to watch for over-editing artefacts.
3. **CLIP score and LPIPS are interfaces, not implementations.** `clip_score` takes injected
   embeddings; no CLIP or `lpips` package is installed. The report labels what it measured
   rather than calling an unrelated proxy a CLIP score.
4. **Region IoU is threshold-sensitive.** The regional case shows no IoU gain (0.422 across
   all arms) even though leakage drops 0.578 -> 0.080, because residual change still clears
   the 5%-of-peak detection threshold. Leakage is the more trustworthy headline.
5. **Residual leakage on local edits (0.35-0.57) is expected**, from the feathered mask edge
   and the deliberately delayed blend ramp. Tightening `feather_radius` or `ramp` trades
   seam quality against leakage.
6. **Attention masking is verified against real diffusers attention, not a checkpoint.**
   `RegionAwareAttnProcessor` is exercised against real
   `diffusers.models.attention_processor.Attention` modules at batch 1/2/4 (this is what
   caught the `prepare_attention_mask` batch bug), but nothing has yet called
   `transformer.set_attn_processor` on loaded PixArt weights.
7. **`locate_edit_tokens` without a tokenizer is word-position based**, which is wrong for
   sub-word encoders like T5. Pass the pipeline's tokenizer: with one, `map_pieces_to_words`
   segments pieces by the tokenizer's own scheme and both roles and token indices align
   per sub-word.
8. **`context_boost` is unvalidated** - see the tuning section. The eval can price its cost
   but not its benefit.
9. **Scene facts are an input, not an inference.** The count-conflict check needs a detector
   or captioner to supply `{"person": 3}`; without it that check is skipped rather than
   guessed at.

**Next steps, in order:** attach the processor to a real PixArt transformer and confirm the
bias applies at every cross-attention block → re-run the eval against the checkpoint on a
CUDA worker → recalibrate `similarity_floor/ceiling` for the actual text encoder → tune
`locality_weight`/`similarity_weight` (exposed as eval CLI flags) → then add the img2img
entry point to `ModelService` and the API.

## Tuning

| Parameter | Effect | Where |
|---|---|---|
| `locality_weight` / `similarity_weight` | balance of mask size vs prompt conflict | `CoefficientConfig` |
| `similarity_floor` / `similarity_ceiling` | encoder calibration band - set these first | `CoefficientConfig` |
| `inside_gain` / `outside_damp` | how far the spatial map departs from the base scale | `CoefficientConfig` |
| `min/max_ref_weight` | preservation clamp | `CoefficientConfig` |
| `attention_strength` | 0 disables masking, 1 is effectively a hard mask; graded between | `EditPlan` |
| `leak_penalty` | how hard edit tokens are suppressed outside the region | `build_attention_bias` |
| `context_boost` | pull of context tokens inside the region (see caveat above) | `build_attention_bias` |
| `threshold` | soft->binary cut when inferring the mask from attention | `extract_edit_mask` |
| `blend_width` | feather radius of the edge composite | `apply_edge_blending` |
| `feather_radius` / `dilate_radius` | seam softness vs leakage | `plan_edit` |
| `ramp` | how late blending locks the frame | `preservation_at_step` |

```bash
python scripts/eval_editing.py --locality-weight 0.7 --similarity-weight 0.3
python scripts/eval_editing.py --sweep  # grid-search leak_penalty x context_boost
python scripts/eval_editing.py --json   # machine-readable, for sweeps
```
