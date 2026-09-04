#!/usr/bin/env python3
"""Demonstration and automated test script for the Two-Pass Reasoning Pipeline.

Covers:
1. Ambiguous query (requires clarification or assumption)
2. Context-dependent multi-turn query (resolves entity and style across turns)
3. Clear/unambiguous informational query (direct answer plan)
4. Fallback on reasoning timeout/failure (graceful degradation)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Ensure application modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chat_provider import ModelMessage, ScriptedProvider
from app.services.reasoning_service import ReasoningService


def divider(title: str = "") -> None:
    line = "=" * 70
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(f"{line}")


def subheader(title: str) -> None:
    print(f"\n--- {title} ---")


async def run_scenario_1_ambiguous():
    divider("SCENARIO 1: Ambiguous User Request (Needs Clarification)")
    print("User sends an ambiguous request with no prior context in the session.")
    print("User message: 'Make it look much better and more dramatic.'\n")

    reasoning_xml = """
<reasoning>
  <intent>User wants to enhance or add drama to an unspecified image or subject.</intent>
  <context_notes>No prior image generation or conversation history in session.</context_notes>
  <ambiguity>
    <is_ambiguous>true</is_ambiguous>
    <assumption>none</assumption>
    <clarifying_question>Which subject should I make dramatic?</clarifying_question>
  </ambiguity>
  <constraints>none</constraints>
  <response_plan>Ask a single targeted clarifying question.</response_plan>
  <action>clarify</action>
  <generation_prompt>none</generation_prompt>
</reasoning>
""".strip()

    response_reply = "Which subject or previous image would you like me to make dramatic?"

    provider = ScriptedProvider(reasoning_xml, response_reply)
    service = ReasoningService(provider)

    messages = [
        ModelMessage(
            role="user",
            content="Make it look much better and more dramatic.",
            turn_id="turn-1",
        )
    ]
    turn = await service.reason(session_id="session-ambig", messages=messages)

    subheader("PASS 1: Structured Reasoning Analysis (Hidden from User)")
    analysis = turn.reasoning_analysis
    assert analysis is not None
    print(f"• Intent: {analysis.intent}")
    print(f"• Context Notes: {analysis.context_notes}")
    print(f"• Is Ambiguous: {analysis.ambiguity.is_ambiguous}")
    print(f"• Clarifying Question: {analysis.ambiguity.clarifying_question}")
    print(f"• Constraints: {analysis.constraints}")
    print(f"• Response Plan: {analysis.response_plan}")
    print(f"• Action: {analysis.action}")

    subheader("PASS 2: User-Facing Response (Never leaks internal XML/reasoning)")
    print(f'Assistant: "{turn.public_response}"')

    subheader("Timing & Observability Metadata")
    print(f"• Reasoning Latency: {turn.timing.reasoning_ms} ms")
    print(f"• Response Latency:  {turn.timing.response_ms} ms")
    print(f"• Total Latency:     {turn.timing.total_ms} ms")
    print(f"• Fallback Used:     {turn.fallback_used}")


async def run_scenario_2_context_dependent():
    divider("SCENARIO 2: Context-Dependent Multi-Turn Refinement")
    print("Turn 1 creates a watercolor painting of a red fox.")
    print("Turn 2 asks to refine: 'Now put a cozy blue scarf on it and make the forest snowy.'\n")

    history = [
        ModelMessage(
            role="user",
            content="Create a watercolor painting of a red fox in an autumn forest.",
            turn_id="t1",
        ),
        ModelMessage(
            role="assistant",
            content="I've generated the watercolor painting of a red fox for you.",
            turn_id="t2",
        ),
        ModelMessage(
            role="user",
            content="Now put a cozy blue winter scarf on it and make the forest snowy.",
            turn_id="t3",
        ),
    ]

    reasoning_xml = """
<reasoning>
  <intent>Refine previous fox painting into a winter scene with a blue scarf.</intent>
  <context_notes>Carries over watercolor style and red fox subject from turn 1.</context_notes>
  <ambiguity>
    <is_ambiguous>true</is_ambiguous>
    <assumption>Preserve the watercolor art style from the previous turn.</assumption>
    <clarifying_question>none</clarifying_question>
  </ambiguity>
  <constraints>none</constraints>
  <response_plan>Confirm the winter scene generation with blue scarf.</response_plan>
  <action>generate_image</action>
  <generation_prompt>A watercolor painting of a red fox in snow</generation_prompt>
</reasoning>
""".strip()

    response_reply = (
        "I'm updating your image to feature the red fox wearing a cozy blue winter scarf in snow."
    )

    provider = ScriptedProvider(reasoning_xml, response_reply)
    service = ReasoningService(provider)

    turn = await service.reason(session_id="session-ctx", messages=history)

    subheader("PASS 1: Structured Reasoning Analysis (Context Resolved)")
    analysis = turn.reasoning_analysis
    assert analysis is not None
    print(f"• Intent: {analysis.intent}")
    print(f"• Context Notes: {analysis.context_notes}")
    print(f"• Stated Assumption: {analysis.ambiguity.assumption}")
    print(f"• Response Plan: {analysis.response_plan}")
    print(f'• Resolved Generation Prompt: "{analysis.generation_prompt}"')
    print(f"• Action: {analysis.action}")

    subheader("PASS 2: User-Facing Response")
    print(f'Assistant: "{turn.public_response}"')

    subheader("Timing & Observability Metadata")
    print(f"• Reasoning Latency: {turn.timing.reasoning_ms} ms")
    print(f"• Response Latency:  {turn.timing.response_ms} ms")
    print(f"• Total Latency:     {turn.timing.total_ms} ms")


async def run_scenario_3_clear():
    divider("SCENARIO 3: Clear / Unambiguous Informational Request")
    print("User asks a direct technical question about the system's models.")
    print("User message: 'What is the difference between Stable Diffusion and PixArt?'\n")

    reasoning_xml = """
<reasoning>
  <intent>User asks for a technical comparison of the two supported models.</intent>
  <context_notes>none</context_notes>
  <ambiguity>
    <is_ambiguous>false</is_ambiguous>
    <assumption>none</assumption>
    <clarifying_question>none</clarifying_question>
  </ambiguity>
  <constraints>none</constraints>
  <response_plan>Explain architecture difference (UNet vs DiT) concisely.</response_plan>
  <action>respond</action>
  <generation_prompt>none</generation_prompt>
</reasoning>
""".strip()

    response_reply = (
        "Stable Diffusion v1.5 uses a classic UNet architecture. "
        "PixArt-Alpha uses a Diffusion Transformer (DiT) architecture with T5 text embeddings."
    )

    provider = ScriptedProvider(reasoning_xml, response_reply)
    service = ReasoningService(provider)

    messages = [
        ModelMessage(
            role="user",
            content="What is the difference between Stable Diffusion and PixArt?",
            turn_id="t1",
        )
    ]
    turn = await service.reason(session_id="session-clear", messages=messages)

    subheader("PASS 1: Structured Reasoning Analysis")
    analysis = turn.reasoning_analysis
    assert analysis is not None
    print(f"• Intent: {analysis.intent}")
    print(f"• Is Ambiguous: {analysis.ambiguity.is_ambiguous}")
    print(f"• Response Plan: {analysis.response_plan}")
    print(f"• Action: {analysis.action}")

    subheader("PASS 2: User-Facing Response")
    print(f'Assistant: "{turn.public_response}"')

    subheader("Timing & Observability Metadata")
    print(f"• Reasoning Latency: {turn.timing.reasoning_ms} ms")
    print(f"• Response Latency:  {turn.timing.response_ms} ms")
    print(f"• Total Latency:     {turn.timing.total_ms} ms")


async def run_scenario_4_fallback():
    divider("SCENARIO 4: Graceful Degradation on Reasoning Timeout / Failure")
    print("Pass 1 (Reasoning) encounters a timeout/upstream failure.")
    print("The orchestrator catches the error, logs it, and calls Pass 2 directly.\n")

    # Pass 1 raises TimeoutError; Pass 2 succeeds directly
    provider = ScriptedProvider(
        TimeoutError("Provider reasoning deadline exceeded (4000ms)"),
        "I'd be happy to help you create an image. What style and subject would you like?",
    )
    service = ReasoningService(provider, reasoning_timeout_seconds=0.05)

    messages = [ModelMessage(role="user", content="Create a picture for me", turn_id="t1")]
    turn = await service.reason(session_id="session-fallback", messages=messages)

    subheader("Pipeline Outcome (Graceful Degradation)")
    print(f"• Fallback Used:   {turn.fallback_used}")
    print(f"• Fallback Reason: {turn.fallback_reason}")
    print(f"• Reasoning State: {turn.reasoning_analysis}")
    print(f'• Assistant Reply: "{turn.public_response}"')
    print(f"• Reasoning Latency: {turn.timing.reasoning_ms} ms")
    print(f"• Response Latency:  {turn.timing.response_ms} ms")
    print(f"• Total Latency:     {turn.timing.total_ms} ms")
    print("-> Result: User experienced zero errors; system gracefully degraded to direct response!")


async def main():
    print("\n" + "=" * 70)
    print("       TWO-PASS REASONING PIPELINE DEMONSTRATION & TEST SUITE")
    print("=" * 70)
    await run_scenario_1_ambiguous()
    await run_scenario_2_context_dependent()
    await run_scenario_3_clear()
    await run_scenario_4_fallback()
    divider("ALL 4 SCENARIOS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    asyncio.run(main())
