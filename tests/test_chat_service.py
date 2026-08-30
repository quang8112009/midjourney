import asyncio
import unittest

from app.services.chat_service import ChatService
from app.services.conversation_store import ConversationStore
from app.services.reasoning_service import ReasonedTurn


def reasoned(
    *,
    action="respond",
    response="Done.",
    generation_prompt=None,
    trace_id="trace-1",
):
    return ReasonedTurn(
        trace_id=trace_id,
        action=action,
        public_response=response,
        generation_prompt=generation_prompt,
        fallback_used=False,
        fallback_reason=None,
    )


class ScriptedReasoner:
    def __init__(self, *results, delay=0):
        self.results = list(results)
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def reason(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.results.pop(0)
        finally:
            self.active -= 1

    async def aclose(self):
        return None


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_turn_commits_public_exchange_and_effective_prompt(self):
        store = ConversationStore()
        reasoner = ScriptedReasoner(
            reasoned(
                action="generate_image",
                response="Creating that image.",
                generation_prompt="a white sneaker on a neutral background",
            )
        )
        service = ChatService(store=store, reasoning_service=reasoner)
        generated = object()

        async def generate(prompt):
            self.assertEqual(prompt, "a white sneaker on a neutral background")
            return generated, f"{prompt}, warm studio lighting"

        outcome = await service.handle_turn(
            session_id="session-1",
            message="  Create a sneaker product shot  ",
            generate=generate,
        )

        self.assertIs(outcome.generation, generated)
        self.assertEqual(outcome.user_turn.content, "Create a sneaker product shot")
        self.assertEqual(outcome.assistant_turn.content, "Creating that image.")
        snapshot = store.get_snapshot("session-1")
        self.assertEqual([turn.role for turn in snapshot.turns], ["user", "assistant"])
        self.assertEqual(
            snapshot.last_generation_prompt,
            "a white sneaker on a neutral background, warm studio lighting",
        )

    async def test_next_turn_receives_ordered_public_history_only(self):
        store = ConversationStore()
        reasoner = ScriptedReasoner(
            reasoned(response="First answer.", trace_id="trace-1"),
            reasoned(response="Second answer.", trace_id="trace-2"),
        )
        service = ChatService(store=store, reasoning_service=reasoner)

        first = await service.handle_turn(session_id="session", message="First question")
        await service.handle_turn(session_id="session", message="Follow up")

        second_messages = reasoner.calls[1]["messages"]
        self.assertEqual(
            [(message.role, message.content) for message in second_messages],
            [
                ("user", "First question"),
                ("assistant", "First answer."),
                ("user", "Follow up"),
            ],
        )
        self.assertEqual(second_messages[0].turn_id, first.user_turn.turn_id)
        self.assertFalse(hasattr(store.get_snapshot("session"), "reasoning"))

    async def test_refinement_receives_last_effective_prompt_as_labeled_context(self):
        store = ConversationStore()
        reasoner = ScriptedReasoner(
            reasoned(
                action="generate_image",
                response="Created the sneaker.",
                generation_prompt="a white sneaker",
                trace_id="trace-1",
            ),
            reasoned(response="Making it warmer.", trace_id="trace-2"),
        )
        service = ChatService(store=store, reasoning_service=reasoner)

        async def generate(prompt):
            return object(), f"{prompt}, neutral background"

        await service.handle_turn(
            session_id="session",
            message="Create a white sneaker",
            generate=generate,
        )
        await service.handle_turn(
            session_id="session",
            message="Make it warmer",
            generate=generate,
        )

        context = reasoner.calls[1]["messages"][-2]
        self.assertEqual(context.role, "assistant")
        self.assertEqual(context.kind, "generation_context")
        self.assertEqual(context.content, "a white sneaker, neutral background")
        self.assertTrue(context.turn_id.endswith(":generation-prompt"))

    async def test_failed_generation_does_not_commit_success_or_prompt(self):
        store = ConversationStore()
        reasoner = ScriptedReasoner(
            reasoned(
                action="generate_image",
                response="Creating it.",
                generation_prompt="a castle",
            )
        )
        service = ChatService(store=store, reasoning_service=reasoner)

        async def fail_generation(_prompt):
            raise RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            await service.handle_turn(
                session_id="session",
                message="Create a castle",
                generate=fail_generation,
            )

        self.assertEqual(store.get_history("session"), ())
        self.assertIsNone(store.get_last_generation_prompt("session"))

    async def test_same_session_is_serialized_but_different_sessions_overlap(self):
        same_reasoner = ScriptedReasoner(
            reasoned(response="one", trace_id="one"),
            reasoned(response="two", trace_id="two"),
            delay=0.02,
        )
        same_service = ChatService(
            store=ConversationStore(),
            reasoning_service=same_reasoner,
        )
        await asyncio.gather(
            same_service.handle_turn(session_id="same", message="one"),
            same_service.handle_turn(session_id="same", message="two"),
        )
        self.assertEqual(same_reasoner.max_active, 1)
        self.assertEqual(len(same_reasoner.calls[1]["messages"]), 3)

        different_reasoner = ScriptedReasoner(
            reasoned(response="one", trace_id="one"),
            reasoned(response="two", trace_id="two"),
            delay=0.02,
        )
        different_service = ChatService(
            store=ConversationStore(),
            reasoning_service=different_reasoner,
        )
        await asyncio.gather(
            different_service.handle_turn(session_id="a", message="one"),
            different_service.handle_turn(session_id="b", message="two"),
        )
        self.assertEqual(different_reasoner.max_active, 2)


if __name__ == "__main__":
    unittest.main()
