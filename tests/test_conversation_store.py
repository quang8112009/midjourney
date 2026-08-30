import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields

from app.services.conversation_store import (
    ConversationSnapshot,
    ConversationStore,
    ConversationTurn,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.value

    def advance(self, seconds):
        with self._lock:
            self.value += seconds


class ConversationStoreTests(unittest.TestCase):
    def test_turns_and_snapshots_are_immutable_public_data_only(self):
        store = ConversationStore()
        turn = store.append_turn(
            "session-1",
            role="user",
            content="  Draw a red kite  ",
            turn_id="turn-1",
        )
        store.set_last_generation_prompt("session-1", "A detailed red kite")

        snapshot = store.get_snapshot("session-1")

        self.assertEqual(turn.content, "Draw a red kite")
        self.assertEqual(snapshot.turns, (turn,))
        self.assertEqual(snapshot.last_generation_prompt, "A detailed red kite")
        self.assertEqual(
            {field.name for field in fields(ConversationTurn)},
            {"turn_id", "role", "content"},
        )
        self.assertEqual(
            {field.name for field in fields(ConversationSnapshot)},
            {"session_id", "turns", "last_generation_prompt"},
        )
        with self.assertRaises(FrozenInstanceError):
            turn.content = "hidden reasoning"
        with self.assertRaises(FrozenInstanceError):
            snapshot.last_generation_prompt = "replacement"

    def test_message_and_character_limits_evict_oldest_turns(self):
        store = ConversationStore(
            max_messages_per_session=3,
            max_history_chars=9,
        )
        store.append_turn("session", role="user", content="aa", turn_id="a")
        store.append_turn("session", role="assistant", content="bbb", turn_id="b")
        store.append_turn("session", role="user", content="cccc", turn_id="c")
        store.append_turn("session", role="assistant", content="dd", turn_id="d")

        self.assertEqual(
            [turn.turn_id for turn in store.get_history("session")],
            ["c", "d"],
        )

        store.append_turn("session", role="user", content="e", turn_id="e")
        self.assertEqual(
            [turn.turn_id for turn in store.get_history("session")],
            ["c", "d", "e"],
        )

    def test_atomic_exchange_eviction_never_exposes_orphan_assistant(self):
        store = ConversationStore(max_messages_per_session=4, max_history_chars=20)
        store.append_exchange(
            "session",
            user_content="one",
            assistant_content="first",
            user_turn_id="u1",
            assistant_turn_id="a1",
        )
        store.append_exchange(
            "session",
            user_content="two",
            assistant_content="second",
            user_turn_id="u2",
            assistant_turn_id="a2",
            last_generation_prompt="image two",
        )
        store.append_exchange(
            "session",
            user_content="three",
            assistant_content="third",
            user_turn_id="u3",
            assistant_turn_id="a3",
        )

        history = store.get_history("session")
        self.assertEqual([turn.turn_id for turn in history], ["u2", "a2", "u3", "a3"])
        self.assertEqual(history[0].role, "user")
        self.assertEqual(store.get_last_generation_prompt("session"), "image two")

    def test_oversized_content_is_rejected_without_mutating_history(self):
        store = ConversationStore(max_history_chars=4)
        original = store.append_turn("session", role="user", content="four")

        with self.assertRaisesRegex(ValueError, "max_history_chars"):
            store.append_turn("session", role="assistant", content="12345")

        self.assertEqual(store.get_history("session"), (original,))

    def test_max_sessions_uses_lru_eviction(self):
        clock = FakeClock()
        store = ConversationStore(max_sessions=2, clock=clock)
        store.append_turn("old", role="user", content="old")
        clock.advance(1)
        store.append_turn("recent", role="user", content="recent")
        clock.advance(1)
        self.assertEqual(len(store.get_history("old")), 1)
        clock.advance(1)

        store.append_turn("new", role="user", content="new")

        self.assertEqual(len(store.get_history("old")), 1)
        self.assertEqual(store.get_history("recent"), ())
        self.assertEqual(len(store.get_history("new")), 1)
        self.assertEqual(store.session_count, 2)

    def test_ttl_uses_injected_clock_and_access_extends_session(self):
        clock = FakeClock()
        store = ConversationStore(ttl_seconds=10, clock=clock)
        store.append_turn("active", role="user", content="hello")
        store.append_turn("idle", role="user", content="goodbye")

        clock.advance(9)
        self.assertEqual(len(store.get_history("active")), 1)
        clock.advance(1)
        self.assertEqual(store.purge_expired(), 1)
        self.assertEqual(store.get_history("idle"), ())
        self.assertEqual(len(store.get_history("active")), 1)
        clock.advance(10)
        self.assertEqual(store.get_history("active"), ())

    def test_last_generation_prompt_is_bounded_clearable_and_expires(self):
        clock = FakeClock()
        store = ConversationStore(max_history_chars=8, ttl_seconds=5, clock=clock)

        store.set_last_generation_prompt("session", "  a castle  ")
        self.assertEqual(store.get_last_generation_prompt("session"), "a castle")
        with self.assertRaisesRegex(ValueError, "max_history_chars"):
            store.set_last_generation_prompt("session", "long castle")
        self.assertEqual(store.get_last_generation_prompt("session"), "a castle")

        store.set_last_generation_prompt("session", None)
        self.assertIsNone(store.get_last_generation_prompt("session"))
        store.set_last_generation_prompt("session", "a castle")
        clock.advance(5)
        self.assertIsNone(store.get_snapshot("session"))

    def test_duplicate_turn_ids_and_invalid_input_are_rejected(self):
        store = ConversationStore()
        store.append_turn("session", role="user", content="one", turn_id="duplicate")
        with self.assertRaisesRegex(ValueError, "already exists"):
            store.append_turn("session", role="assistant", content="two", turn_id="duplicate")
        with self.assertRaisesRegex(ValueError, "role"):
            store.append_turn("session", role="system", content="hidden")
        with self.assertRaisesRegex(ValueError, "blank"):
            store.append_turn("session", role="user", content="  ")

    def test_concurrent_appends_do_not_lose_or_duplicate_turns(self):
        store = ConversationStore(
            max_messages_per_session=250,
            max_history_chars=10_000,
        )

        def append(index):
            return store.append_turn(
                "shared-session",
                role="user" if index % 2 == 0 else "assistant",
                content=f"message-{index}",
                turn_id=f"turn-{index}",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            turns = list(executor.map(append, range(200)))

        history = store.get_history("shared-session")
        self.assertEqual(len(history), 200)
        self.assertEqual(len({turn.turn_id for turn in history}), 200)
        self.assertEqual({turn.turn_id for turn in history}, {turn.turn_id for turn in turns})

    def test_constructor_bounds_and_session_management(self):
        invalid_options = (
            {"max_sessions": 0},
            {"max_messages_per_session": 1},
            {"max_history_chars": 0},
            {"ttl_seconds": 0},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                ConversationStore(**options)

        store = ConversationStore()
        generated_id = store.create_session()
        self.assertTrue(generated_id)
        self.assertEqual(store.session_count, 1)
        self.assertTrue(store.delete_session(generated_id))
        self.assertFalse(store.delete_session(generated_id))
        store.create_session("a")
        store.create_session("b")
        store.clear()
        self.assertEqual(store.session_count, 0)


if __name__ == "__main__":
    unittest.main()
