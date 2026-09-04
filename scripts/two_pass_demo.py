#!/usr/bin/env python3
"""Offline walkthrough of the two-pass reasoning pipeline.

Runs entirely against a scripted in-process provider: no API key, no network,
no diffusion model. Each scenario prints the hidden Pass 1 analysis, the public
Pass 2 reply, and the per-pass timings.

    python scripts/two_pass_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chat_provider import ModelMessage  # noqa: E402
from app.services.reasoning_service import ReasoningService  # noqa: E402


def analysis_xml(
    *,
    intent,
    response_plan,
    context_notes="none",
    is_ambiguous=False,
    assumption=None,
    clarifying_question=None,
    constraints="none",
    action="respond",
    generation_prompt=None,
):
    """Build a Pass 1 payload in the XML contract the analysis prompt asks for."""
    parts = [
        f"<intent>{intent}</intent>",
        f"<context_notes>{context_notes}</context_notes>",
        "<ambiguity>",
        f"  <is_ambiguous>{str(is_ambiguous).lower()}</is_ambiguous>",
    ]
    if assumption:
        parts.append(f"  <assumption>{assumption}</assumption>")
    if clarifying_question:
        parts.append(f"  <clarifying_question>{clarifying_question}</clarifying_question>")
    parts += [
        "</ambiguity>",
        f"<constraints>{constraints}</constraints>",
        f"<response_plan>{response_plan}</response_plan>",
        f"<action>{action}</action>",
    ]
    if generation_prompt:
        parts.append(f"<generation_prompt>{generation_prompt}</generation_prompt>")
    return "<reasoning>\n" + "\n".join(parts) + "\n</reasoning>"


class ScriptedProvider:
    """Returns canned outputs in order, standing in for a real model."""

    provider_name = "scripted-demo"
    model_name = "scripted-demo-model"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise RuntimeError("Scripted provider exhausted")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output

    async def aclose(self):
        return None


class HangingProvider(ScriptedProvider):
    """Blocks past the analysis deadline to prove the timeout path degrades."""

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            await asyncio.sleep(5)
            raise AssertionError("unreachable: analysis should have timed out")
        return self.outputs.pop(0)


def turns(*pairs):
    """Build ordered conversation history ending on the current user turn."""
    return [
        ModelMessage(role=role, content=content, turn_id=f"t{index}")
        for index, (role, content) in enumerate(pairs, start=1)
    ]


SCENARIOS = [
    (
        "1. AMBIGUOUS — no referent for 'it'",
        ScriptedProvider(
            analysis_xml(
                intent="Refine an image the user has not identified",
                is_ambiguous=True,
                clarifying_question="Which image would you like me to make more dramatic?",
                response_plan="Ask which image is meant; do not guess",
                action="clarify",
            ),
            "Which image would you like me to make more dramatic?",
        ),
        turns(("user", "Make it more dramatic.")),
    ),
    (
        "2. CONTEXT-DEPENDENT — 'warmer' refers to an earlier turn",
        ScriptedProvider(
            analysis_xml(
                intent="Warm the lighting of the existing sneaker product shot",
                context_notes="Turn t1 established a white sneaker on a neutral background",
                response_plan="Confirm the warmer lighting and that the background is unchanged",
                action="generate_image",
                generation_prompt=(
                    "Product shot of a white sneaker on a neutral background, "
                    "warm studio lighting, soft golden key light"
                ),
            ),
            "Warming up the lighting while keeping that neutral background.",
        ),
        turns(
            ("user", "Create a product shot of a white sneaker on a neutral background."),
            ("assistant", "Creating the product shot."),
            ("user", "Make it warmer but keep the background neutral."),
        ),
    ),
    (
        "3. CLEAR — unambiguous request, no history needed",
        ScriptedProvider(
            analysis_xml(
                intent="Generate a red fox in a snowy forest",
                response_plan="Confirm generation in one short sentence",
                action="generate_image",
                generation_prompt="A red fox in a snowy forest, soft winter light",
            ),
            "Creating your red fox in a snowy forest now.",
        ),
        turns(("user", "Create a red fox in a snowy forest.")),
    ),
    (
        "4. ANALYSIS FAILURE — Pass 1 times out, user still gets a clean reply",
        HangingProvider("Stable Diffusion runs on CPU, just more slowly."),
        turns(("user", "Can Stable Diffusion run on CPU?")),
    ),
]


def show(label, value, indent="      "):
    """Print one analysis field, skipping the ones the model left empty."""
    if value in (None, "", "none"):
        return
    wrapped = textwrap.fill(str(value), width=88, subsequent_indent=indent + " " * 18).strip()
    print(f"{indent}{label:<16} {wrapped}")


async def main():
    for title, provider, messages in SCENARIOS:
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)
        print(f"  User: {messages[-1].content}")
        if len(messages) > 1:
            print(f"  (history: {len(messages) - 1} prior turn(s))")

        service = ReasoningService(provider, reasoning_timeout_seconds=0.5)
        turn = await service.reason(session_id="demo-session", messages=messages)

        print("\n  --- PASS 1: hidden analysis (never shown to the user) ---")
        analysis = turn.reasoning_analysis
        if analysis is None:
            print(f"      <unavailable: {turn.fallback_reason}>")
            print("      Pipeline degraded to a direct reply with no analysis context.")
        else:
            show("intent:", analysis.intent)
            show("context_notes:", analysis.context_notes)
            show("is_ambiguous:", analysis.ambiguity.is_ambiguous)
            show("assumption:", analysis.ambiguity.assumption)
            show("clarifying_q:", analysis.ambiguity.clarifying_question)
            show("constraints:", analysis.constraints)
            show("response_plan:", analysis.response_plan)
            show("action:", analysis.action)
            show("gen_prompt:", analysis.generation_prompt)

        print("\n  --- PASS 2: public reply ---")
        print(
            textwrap.fill(
                turn.public_response, width=88, initial_indent="      ", subsequent_indent="      "
            )
        )

        t = turn.timing
        reasoning = "n/a" if t.reasoning_ms is None else f"{t.reasoning_ms:.1f}ms"
        skipped = " (skipped)" if t.response_ms == 0.0 else ""
        print(
            f"\n  timings: reasoning={reasoning}  response={t.response_ms:.1f}ms{skipped}  "
            f"total={t.total_ms:.1f}ms   provider_calls={len(provider.calls)}   "
            f"action={turn.action}   fallback={turn.fallback_used}"
        )

    print("\n" + "=" * 92)
    print("All scenarios completed offline. No API key or model download was required.")
    print("=" * 92)


if __name__ == "__main__":
    asyncio.run(main())
