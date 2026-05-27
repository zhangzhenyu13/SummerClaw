"""Tests for MastraOMStore — file I/O for OBSERVATIONS.md, history, etc."""

import pytest

from summerclaw.memory.mastra_om_memory.store import MastraOMStore


@pytest.fixture
def store(tmp_path):
    return MastraOMStore(tmp_path)


class TestMastraOMStoreObservations:
    """Tests for observation log I/O."""

    def test_read_empty_observations(self, store):
        assert store.read_observations() == ""

    def test_write_and_read_observations(self, store):
        content = "Date: May 9, 2025\n* 🔴 User prefers dark mode"
        store.write_observations(content)
        assert store.read_observations() == content

    def test_append_observations_to_empty(self, store):
        result = store.append_observations("Date: May 9\n* 🔴 User likes Python")
        assert result is not None
        content = store.read_observations()
        assert "Observational Memory" in content
        assert "User likes Python" in content
        assert "Observation Cycle" in content

    def test_append_observations_to_existing(self, store):
        store.write_observations("# Observational Memory\nOld content")
        store.append_observations("Date: May 10\n* 🟡 New info")
        content = store.read_observations()
        assert "Old content" in content
        assert "New info" in content

    def test_replace_observations(self, store):
        store.write_observations("Old content")
        store.replace_observations("New content")
        assert store.read_observations() == "New content"

    def test_generation_tracking(self, store):
        assert store.get_generation_count() == 0
        assert store.increment_generation() == 1
        assert store.get_generation_count() == 1
        assert store.increment_generation() == 2
        assert store.get_generation_count() == 2


class TestMastraOMStoreHistory:
    """Tests for history.jsonl I/O."""

    def test_append_history_returns_cursor(self, store):
        cursor = store.append_history("First entry")
        assert cursor == 1
        cursor2 = store.append_history("Second entry")
        assert cursor2 == 2

    def test_read_unprocessed_history_returns_empty(self, store):
        """mastra_om always returns empty list — obs replaces raw history injection."""
        store.append_history("Entry 1")
        store.append_history("Entry 2")
        store.append_history("Entry 3")
        assert store.read_unprocessed_history(since_cursor=0) == []
        assert store.read_unprocessed_history(since_cursor=1) == []
        assert store.read_unprocessed_history(since_cursor=100) == []

    def test_compact_history(self, store):
        store.max_history_entries = 3
        for i in range(10):
            store.append_history(f"Entry {i}")
        # compact is not auto-triggered; must call explicitly
        store.compact_history()
        entries = store._read_entries()
        assert len(entries) <= 3


class TestMastraOMStoreSoulUser:
    """Tests for SOUL.md and USER.md I/O."""

    def test_soul_read_write(self, store):
        assert store.read_soul() == ""
        store.write_soul("# Soul\n- Helpful")
        assert "Helpful" in store.read_soul()

    def test_user_read_write(self, store):
        assert store.read_user() == ""
        store.write_user("# User\n- Developer")
        assert "Developer" in store.read_user()


class TestMastraOMStoreMemory:
    """Tests for MEMORY.md I/O."""

    def test_memory_read_write(self, store):
        assert store.read_memory() == ""
        store.write_memory("# Memory\n- Project X")
        assert "Project X" in store.read_memory()


class TestMastraOMStoreCursors:
    """Tests for cursor management."""

    def test_dream_cursor(self, store):
        assert store.get_last_dream_cursor() == 0
        store.set_last_dream_cursor(42)
        assert store.get_last_dream_cursor() == 42

    def test_obs_cursor(self, store):
        assert store.get_last_obs_cursor() == 0
        store.set_last_obs_cursor(100)
        assert store.get_last_obs_cursor() == 100

    def test_cursor_persists(self, tmp_path):
        s1 = MastraOMStore(tmp_path)
        s1.set_last_dream_cursor(10)
        s1.set_last_obs_cursor(20)

        s2 = MastraOMStore(tmp_path)
        assert s2.get_last_dream_cursor() == 10
        assert s2.get_last_obs_cursor() == 20


class TestMastraOMStoreContext:
    """Tests for context injection."""

    def test_get_memory_context_empty(self, store):
        ctx = store.get_memory_context()
        assert ctx == ""

    def test_get_memory_context_with_observations(self, store):
        store.write_observations("Date: May 9\n* 🔴 User likes Python")
        ctx = store.get_memory_context()
        assert "User likes Python" in ctx
        assert "Past Conversation Records" in ctx
        assert "higher informational priority" in ctx

    def test_get_memory_context_with_memory(self, store):
        store.write_memory("# Memory\n- Project X")
        ctx = store.get_memory_context()
        assert "Project X" in ctx
        assert "Long-term Memory" in ctx

    def test_get_memory_context_combined(self, store):
        store.write_observations("Date: May 9\n* 🔴 User likes Python")
        store.write_memory("# Memory\n- Project X")
        ctx = store.get_memory_context()
        assert "User likes Python" in ctx
        assert "Project X" in ctx


class TestMastraOMStoreFormatting:
    """Tests for message formatting utility."""

    def test_format_messages(self, store):
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2025-05-09 10:00"},
            {"role": "assistant", "content": "Hi there!", "timestamp": "2025-05-09 10:01"},
        ]
        result = store._format_messages(messages)
        assert "USER" in result
        assert "ASSISTANT" in result
        assert "Hello" in result
        assert "Hi there!" in result

    def test_format_messages_with_tools(self, store):
        messages = [
            {
                "role": "assistant",
                "content": "Done",
                "tools_used": ["read_file", "edit_file"],
                "timestamp": "2025-05-09 10:00",
            },
        ]
        result = store._format_messages(messages)
        assert "[tools:" in result
        assert "read_file" in result


class TestMastraOMStoreRawArchive:
    """Tests for raw_archive fallback."""

    def test_raw_archive(self, store):
        messages = [{"role": "user", "content": "test"}]
        store.raw_archive(messages)
        # read_unprocessed_history returns [] for mastra_om; use _read_entries directly
        entries = store._read_entries()
        assert len(entries) >= 1
        assert "[RAW]" in entries[-1]["content"]


class TestMastraOMStoreDreamGeneration:
    """Tests for dream generation tracking."""

    def test_dream_generation_default_zero(self, store):
        assert store.get_last_dream_generation() == 0

    def test_set_and_get_dream_generation(self, store):
        store.set_last_dream_generation(5)
        assert store.get_last_dream_generation() == 5

    def test_dream_generation_persists(self, tmp_path):
        s1 = MastraOMStore(tmp_path)
        s1.set_last_dream_generation(10)
        s2 = MastraOMStore(tmp_path)
        assert s2.get_last_dream_generation() == 10

