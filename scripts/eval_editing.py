#!/usr/bin/env python3
"""Before/after evaluation of the region-aware editing path.

Runs offline on CPU with no checkpoint. The denoiser is a *simulator*: it models
the one property that matters for this comparison - a DiT applies the prompt
direction to the whole frame, because nothing tells it where the edit belongs.
Both arms call the identical simulator; only the conditioning differs, so the
delta is attributable to the mechanism and not to the harness.

Three arms:
  baseline  - today's behaviour: one scalar guidance_scale, no mask, no blending
  legacy    - the supplied pseudo-code coefficient, to price its sign error
  proposed  - region-aware attention + adaptive coefficient + scheduled blending

    python scripts/eval_editing.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.editing.adaptive_reference import (  # noqa: E402
    CoefficientConfig,
    ReferenceCoefficients,
    legacy_reference_coefficient,
)
from app.services.editing.edit_pipeline import (  # noqa: E402
    run_baseline_edit,
    run_region_aware_edit,
)
from app.services.editing.edit_planner import plan_edit  # noqa: E402
from app.services.editing.masks import as_soft_mask  # noqa: E402
from app.services.editing.metrics import evaluate_edit, inside_alignment  # noqa: E402
from app.services.editing.region_attention import (  # noqa: E402
    build_attention_bias,
    classify_token_roles,
)

LATENT = 64
CHANNELS = 4
BASE_SCALE = 7.5
STEPS = 12


def structured_source(seed: int) -> torch.Tensor:
    """A smooth, structured latent - not white noise, so SSIM is meaningful."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(1, CHANNELS, 8, 8, generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(LATENT, LATENT), mode="bicubic", align_corners=False
    )


def build_prompt_embedding(
    source: torch.Tensor,
    mask: torch.Tensor,
    similarity: float,
    seed: int,
) -> torch.Tensor:
    """A prompt embedding at a known cosine similarity to the masked region."""
    from app.services.editing.adaptive_reference import extract_region_embedding

    region = extract_region_embedding(source, mask).float()
    norm = region.norm()
    if float(norm) == 0.0:
        return torch.randn(CHANNELS, generator=torch.Generator().manual_seed(seed))
    unit = region / norm
    generator = torch.Generator().manual_seed(seed + 7)
    noise = torch.randn(CHANNELS, generator=generator)
    orthogonal = noise - unit * torch.dot(noise, unit)
    orthogonal = orthogonal / orthogonal.norm().clamp_min(1e-8)
    similarity = max(-1.0, min(1.0, similarity))
    return similarity * unit + (1.0 - similarity**2) ** 0.5 * orthogonal


def attention_gate(
    bias: torch.Tensor | None,
    roles: list[str],
    num_image_tokens: int,
    latent_size: tuple[int, int],
) -> torch.Tensor:
    """Per-position share of cross-attention landing on edit_target tokens.

    This is what couples the attention module to the measured outcome: with no bias
    every position gives the edit tokens the same share (so the edit lands
    everywhere - the leak), and the region bias drives that share toward zero
    outside the mask. Tuning `leak_penalty`/`context_boost` therefore changes the
    numbers rather than being asserted.
    """
    targets = [i for i, role in enumerate(roles) if role == "edit_target"]
    if not targets:
        return torch.zeros(1, 1, *latent_size)
    logits = (
        torch.zeros(num_image_tokens, len(roles))
        if bias is None
        else bias.transpose(0, 1).clone()
    )
    share = logits.softmax(dim=-1)[:, targets].sum(dim=-1)
    return share.reshape(1, 1, *latent_size)


def make_simulator(
    source: torch.Tensor,
    direction: torch.Tensor,
    magnitude: float,
    gate: torch.Tensor | None = None,
):
    """A denoiser whose edit lands wherever the edit tokens got attention."""
    weight = 1.0 if gate is None else gate
    target = source + magnitude * direction * weight

    def denoise(latents: torch.Tensor, timestep: int, conditional: bool) -> torch.Tensor:
        return latents - (target if conditional else source)

    def step(latents: torch.Tensor, noise_pred: torch.Tensor, timestep: int) -> torch.Tensor:
        # Guidance is interpreted relative to the base scale and damped per step so
        # the loop converges instead of overshooting.
        return latents - noise_pred / STEPS

    return denoise, step


def normalized_coefficients(coefficients: ReferenceCoefficients) -> ReferenceCoefficients:
    """Express guidance as a multiple of the base scale for the simulator."""
    return ReferenceCoefficients(
        ref_weight=coefficients.ref_weight,
        edit_strength=coefficients.edit_strength,
        locality_score=coefficients.locality_score,
        conflict_score=coefficients.conflict_score,
        raw_similarity=coefficients.raw_similarity,
        inside_scale=coefficients.inside_scale / BASE_SCALE,
        outside_scale=coefficients.outside_scale / BASE_SCALE,
    )


def region_alignment(source, edited, direction, mask) -> float:
    """How faithfully the change inside the region followed the requested direction."""
    return inside_alignment(source, edited, direction, as_soft_mask(mask))


CASES = [
    {
        "name": "local: change the shirt to red",
        "prompt": "change the shirt color to red",
        "similarity": 0.12,
        "mask_box": (24, 24, 40, 40),  # 6.25% of frame
        "magnitude": 1.0,
    },
    {
        "name": "local: small object recolor",
        "prompt": "make the mug blue",
        "similarity": 0.12,
        "mask_box": (8, 44, 18, 54),  # 2.4%
        "magnitude": 1.0,
    },
    {
        "name": "local+context: recolor, preserve background",
        "prompt": "change the jacket to red but keep the background neutral",
        "similarity": 0.12,
        "mask_box": (20, 20, 38, 38),  # 7.9%
        "magnitude": 1.0,
    },
    {
        "name": "regional: change the sky",
        "prompt": "make the sky stormy and overcast",
        "similarity": 0.2,
        "mask_box": (0, 0, 26, 64),  # 40%
        "magnitude": 1.0,
    },
    {
        "name": "global: watercolor restyle",
        "prompt": "make the whole image a watercolor painting",
        "similarity": 0.15,
        "mask_box": None,  # global
        "magnitude": 1.0,
    },
    {
        "name": "ambiguous: underspecified",
        "prompt": "make it more dramatic",
        "similarity": 0.3,
        "mask_box": None,
        "magnitude": 1.0,
    },
    {
        "name": "conflicting: add a 2nd person to a 3-person photo",
        "prompt": "add a second person to the photo",
        "similarity": 0.3,
        "mask_box": None,
        "magnitude": 1.0,
        "scene_facts": {"person": 3},
    },
]


METRIC_FIELDS = (
    "alignment", "edit_magnitude", "leakage", "region_iou",
    "preservation_ssim", "preservation_l1",
)


def mean_metrics(rows) -> dict[str, float]:
    """Mean per field, ignoring NaN.

    A global edit has no outside region, so its preservation metrics are
    undefined rather than zero - averaging NaN in would silently poison the table.
    """
    means: dict[str, float] = {}
    for field in METRIC_FIELDS:
        values = [getattr(row, field) for row in rows]
        values = [v for v in values if v == v]  # drop NaN
        means[field] = sum(values) / len(values) if values else float("nan")
    return means


def fmt(value: float, width: int = 9, places: int = 3) -> str:
    return f"{'n/a':>{width}}" if value != value else f"{value:>{width}.{places}f}"


def build_mask(box) -> torch.Tensor:
    mask = torch.zeros(1, 1, LATENT, LATENT)
    if box is None:
        return torch.ones(1, 1, LATENT, LATENT)
    top, left, bottom, right = box
    mask[..., top:bottom, left:right] = 1.0
    return mask


def run_case(
    case,
    seed: int,
    config: CoefficientConfig,
    *,
    leak_penalty: float = -12.0,
    context_boost: float = 0.5,
):
    source = structured_source(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    direction = torch.randn(1, CHANNELS, 1, 1, generator=generator)
    direction = direction / direction.norm()
    mask = build_mask(case["mask_box"])

    # Embedding stand-ins. The prompt embedding is *constructed* to sit at the
    # case's stated cosine similarity to the masked region, so locality and
    # conflict are both controlled variables rather than draws from noise - with
    # random vectors the similarity term is meaningless and the alignment gate
    # fires at random across seeds.
    image_embedding = source
    prompt_embedding = build_prompt_embedding(
        source, mask, case.get("similarity", 0.30), seed
    )

    plan = plan_edit(
        prompt=case["prompt"],
        prompt_embedding=prompt_embedding,
        source_image_embedding=image_embedding,
        user_mask=mask if case["mask_box"] is not None else None,
        scene_facts=case.get("scene_facts"),
        base_guidance_scale=BASE_SCALE,
        config=config,
        # Real-time mode for the guidance cases: a weak prompt/region match is
        # recorded as an assumption and still generates. Only the dedicated
        # conflict case runs with clarification enabled, so the gate is measured
        # on its own rather than intermittently dropping guidance cases.
        allow_clarification=bool(case.get("scene_facts")),
        latent_size=(LATENT, LATENT),
    )

    if not plan.should_generate:
        # Caught before the denoise loop: no diffusion time is spent at all.
        return plan, None

    roles = [role for _, role in classify_token_roles(case["prompt"])]
    num_tokens = LATENT * LATENT

    # Baseline: no region bias at all - the edit tokens attend everywhere.
    baseline_gate = attention_gate(None, roles, num_tokens, (LATENT, LATENT))
    # Proposed: the role-aware bias built from the SAME mask the coefficients used.
    proposed_bias = (
        build_attention_bias(
            plan.mask, roles,
            leak_penalty=leak_penalty * plan.attention_strength,
            context_boost=context_boost * plan.attention_strength,
            num_image_tokens=num_tokens,
        )
        if plan.attention_strength > 0
        else None
    )
    proposed_gate = attention_gate(proposed_bias, roles, num_tokens, (LATENT, LATENT))

    baseline_denoise, step = make_simulator(source, direction, case["magnitude"], baseline_gate)
    proposed_denoise, _ = make_simulator(source, direction, case["magnitude"], proposed_gate)
    denoise = baseline_denoise
    timesteps = list(range(STEPS))
    initial = source.clone()

    baseline = run_baseline_edit(
        source_latents=source,
        initial_latents=initial.clone(),
        timesteps=timesteps,
        denoise=denoise,
        guidance_scale=1.0,  # scalar, uniform - the current pipeline
        step=step,
    )

    legacy_weight = legacy_reference_coefficient(
        prompt_embedding=prompt_embedding,
        source_image_embedding=image_embedding,
        edit_region_mask=plan.mask,
    )
    legacy_coefficients = ReferenceCoefficients(
        ref_weight=legacy_weight,
        edit_strength=1.0 - legacy_weight,
        locality_score=plan.coefficients.locality_score,
        conflict_score=plan.coefficients.conflict_score,
        raw_similarity=plan.coefficients.raw_similarity,
        inside_scale=(1.0 - legacy_weight) * 2.0,  # the pseudo-code's inside rule
        outside_scale=legacy_weight,
    )
    legacy_plan = dataclasses.replace(plan, coefficients=legacy_coefficients)
    legacy = run_region_aware_edit(
        plan=legacy_plan, source_latents=source, initial_latents=initial.clone(),
        timesteps=timesteps, denoise=denoise, step=step, blend=False,
    )

    proposed_plan = dataclasses.replace(
        plan, coefficients=normalized_coefficients(plan.coefficients)
    )
    proposed = run_region_aware_edit(
        plan=proposed_plan, source_latents=source, initial_latents=initial.clone(),
        timesteps=timesteps, denoise=proposed_denoise, step=step, blend=True,
    )

    results = {}
    for arm, edited in (("baseline", baseline), ("legacy", legacy), ("proposed", proposed)):
        results[arm] = evaluate_edit(
            source=source,
            edited=edited,
            edit_mask=plan.mask,
            alignment=region_alignment(source, edited, direction, plan.mask),
        )
    return plan, results


def sweep_score(config, leak_penalty, context_boost, seeds=3):
    """Mean leakage and in-region edit magnitude over the non-global cases."""
    rows = []
    for case in CASES:
        if case["mask_box"] is None:
            continue  # global cases cannot leak; they carry no tuning signal
        for seed in range(seeds):
            _, results = run_case(
                case, seed * 17 + 3, config,
                leak_penalty=leak_penalty, context_boost=context_boost,
            )
            if results is not None:
                rows.append(results["proposed"])
    leakage = sum(r.leakage for r in rows) / len(rows)
    edit = sum(r.edit_magnitude for r in rows) / len(rows)
    return leakage, edit


def run_sweep(args) -> int:
    """Tune the knobs the design leaves open, against measured outcomes."""
    print("=" * 78)
    print("SWEEP 1: leak_penalty x context_boost   (local + regional cases)")
    print("  leakage must fall, but edit-inside must not collapse with it")
    print("=" * 78)
    config = CoefficientConfig(
        locality_weight=args.locality_weight, similarity_weight=args.similarity_weight
    )
    print(f"  {'leak_penalty':>13}{'ctx_boost':>11}{'leakage↓':>11}{'edit_inside↑':>14}")
    rows = []
    for leak_penalty in (-2.0, -4.0, -8.0, -12.0, -20.0):
        for context_boost in (0.0, 0.5, 1.5):
            leakage, edit = sweep_score(config, leak_penalty, context_boost)
            rows.append((leakage, edit, leak_penalty, context_boost))
            print(f"  {leak_penalty:>13.1f}{context_boost:>11.1f}{leakage:>11.4f}{edit:>14.4f}")

    # Lowest leakage is trivially won by an enormous penalty that also kills the
    # edit, so require the edit to stay within 5% of the strongest observed.
    strongest = max(row[1] for row in rows)
    viable = [row for row in rows if row[1] >= 0.95 * strongest]
    best = min(viable, key=lambda row: row[0])
    print(f"\n  strongest edit observed: {strongest:.4f}; "
          f"{len(viable)}/{len(rows)} settings keep >=95% of it")
    print(f"  best viable: leak_penalty={best[2]}, context_boost={best[3]} "
          f"(leakage={best[0]:.4f}, edit={best[1]:.4f})")

    print("\n" + "=" * 78)
    print("SWEEP 2: locality / similarity split in the reference coefficient")
    print("=" * 78)
    print(f"  {'locality':>9}{'similarity':>12}{'leakage↓':>11}{'edit_inside↑':>14}")
    for locality in (0.4, 0.5, 0.6, 0.7, 0.8):
        similarity = round(1.0 - locality, 2)
        swept = CoefficientConfig(locality_weight=locality, similarity_weight=similarity)
        leakage, edit = sweep_score(swept, args.leak_penalty, args.context_boost)
        print(f"  {locality:>9.1f}{similarity:>12.1f}{leakage:>11.4f}{edit:>14.4f}")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    parser.add_argument("--locality-weight", type=float, default=0.6)
    parser.add_argument("--similarity-weight", type=float, default=0.4)
    parser.add_argument("--leak-penalty", type=float, default=-12.0)
    parser.add_argument("--context-boost", type=float, default=0.5)
    parser.add_argument(
        "--sweep", action="store_true",
        help="grid-search leak_penalty x context_boost and the locality/similarity split",
    )
    args = parser.parse_args()

    if args.sweep:
        return run_sweep(args)

    config = CoefficientConfig(
        locality_weight=args.locality_weight,
        similarity_weight=args.similarity_weight,
    )

    payload = {"cases": [], "config": {"locality_weight": args.locality_weight,
                                       "similarity_weight": args.similarity_weight}}
    aggregate: dict[str, list] = {"baseline": [], "legacy": [], "proposed": []}

    if not args.json:
        print("=" * 100)
        print("Region-aware editing - before/after (simulated denoiser, CPU, no checkpoint)")
        print("=" * 100)

    for case in CASES:
        per_arm: dict[str, list] = {"baseline": [], "legacy": [], "proposed": []}
        plan = None
        blocked = False
        for seed in range(args.seeds):
            plan, results = run_case(
                case, seed * 17 + 3, config,
                leak_penalty=args.leak_penalty, context_boost=args.context_boost,
            )
            if results is None:
                blocked = True
                continue
            for arm, metrics in results.items():
                per_arm[arm].append(metrics)
                aggregate[arm].append(metrics)

        if blocked:
            payload["cases"].append({
                "name": case["name"], "scope": plan.scope,
                "blocked_before_denoise": True, "plan": plan.as_log_dict(),
            })
            if not args.json:
                print(f"\n{case['name']}")
                print(f"  BLOCKED before denoising - {plan.alignment.reason}")
                print(f"  asked: {plan.alignment.clarifying_question}")
                print("  cost: 0 denoise steps (the whole point of the pre-generation check)")
            continue

        means = {arm: mean_metrics(rows) for arm, rows in per_arm.items()}
        payload["cases"].append({
            "name": case["name"],
            "scope": plan.scope,
            "plan": plan.as_log_dict(),
            "means": {arm: {k: round(v, 4) for k, v in vals.items()}
                      for arm, vals in means.items()},
        })

        if not args.json:
            print(f"\n{case['name']}")
            print(f"  scope={plan.scope}  mask_source={plan.mask_source}  "
                  f"ref_weight={plan.coefficients.ref_weight:.3f}  "
                  f"attn_strength={plan.attention_strength:.2f}  "
                  f"alignment_status={plan.alignment.status}")
            print(f"  {'arm':<10}{'align↑':>9}{'edit↑':>9}{'leakage↓':>10}"
                  f"{'IoU↑':>8}{'SSIM_out↑':>11}{'L1_out↓':>10}")
            for arm in ("baseline", "legacy", "proposed"):
                m = means[arm]
                print(f"  {arm:<10}{fmt(m['alignment'])}{fmt(m['edit_magnitude'])}"
                      f"{fmt(m['leakage'], 10)}{fmt(m['region_iou'], 8)}"
                      f"{fmt(m['preservation_ssim'], 11)}{fmt(m['preservation_l1'], 10, 4)}")

    overall = {arm: mean_metrics(rows) for arm, rows in aggregate.items()}
    payload["overall"] = {arm: {k: round(v, 4) for k, v in vals.items()}
                          for arm, vals in overall.items()}

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 100)
        print("OVERALL (mean across all cases and seeds)")
        print(f"  {'arm':<10}{'align↑':>9}{'edit↑':>9}{'leakage↓':>10}"
              f"{'IoU↑':>8}{'SSIM_out↑':>11}{'L1_out↓':>10}")
        for arm in ("baseline", "legacy", "proposed"):
            m = overall[arm]
            print(f"  {arm:<10}{fmt(m['alignment'])}{fmt(m['edit_magnitude'])}"
                  f"{fmt(m['leakage'], 10)}{fmt(m['region_iou'], 8)}"
                  f"{fmt(m['preservation_ssim'], 11)}{fmt(m['preservation_l1'], 10, 4)}")
        b, p = overall["baseline"], overall["proposed"]
        reduction = (b["leakage"] - p["leakage"]) / max(b["leakage"], 1e-9) * 100
        print(f"\n  leakage       {b['leakage']:.3f} -> {p['leakage']:.3f}  "
              f"({reduction:.1f}% reduction)")
        print(f"  SSIM outside  {fmt(b['preservation_ssim'], 5)} ->"
              f"{fmt(p['preservation_ssim'], 6)}")
        print(f"  region IoU    {b['region_iou']:.3f} -> {p['region_iou']:.3f}")
        print(f"  edit inside   {b['edit_magnitude']:.3f} -> {p['edit_magnitude']:.3f}"
              "   (must stay comparable - a 'clean' edit that never happened is not a win)")
        print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
