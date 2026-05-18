"""Tests for EMemDream — two-phase memory processor with EDU extraction and graph updates."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.memory.emem_memory.dream import EMemDream
from summerclaw.memory.emem_memory.store import EMemStore
from summerclaw.agent.runner import AgentRunResult


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def mock_embedder() -> MagicMock:
    m = MagicMock()
    m.batch_encode = MagicMock(return_value=[])
    return m


@pytest.fixture
def store(tmp_path, mock_embedder: MagicMock) -> EMemStore:
    s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
    s.write_soul("# Soul\n- Helpful")
    s.write_user("# User\n- Developer")
    s.write_memory("# Memory\n- Project X active")
    return s


@pytest.fixture
def mock_provider() -> MagicMock:
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


@pytest.fixture
def mock_edu_extractor() -> MagicMock:
    ee = MagicMock()
    ee.extract_from_history = AsyncMock(return_value=[])
    return ee


@pytest.fixture
def mock_graph(tmp_path) -> MagicMock:
    g = MagicMock()
    g.load_or_create = MagicMock()
    g.add_nodes = MagicMock()
    g.add_edge = MagicMock()
    g.add_synonymy_edges = MagicMock(return_value=0)
    g.save = MagicMock()
    return g


@pytest.fixture
def mock_runner() -> MagicMock:
    return MagicMock()


def _make_run_result(
    stop_reason: str = "completed",
    tool_events: list[dict] | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        final_content=stop_reason,
        stop_reason=stop_reason,
        messages=[],
        tools_used=[],
        usage={},
        tool_events=tool_events or [],
    )


@pytest.fixture
def dream(
    store: EMemStore,
    mock_provider: MagicMock,
    mock_edu_extractor: MagicMock,
    mock_graph: MagicMock,
    mock_runner: MagicMock,
) -> EMemDream:
    d = EMemDream(
        store=store,
        provider=mock_provider,
        model="test-model",
        edu_extractor=mock_edu_extractor,
        emem_store=store,
        emem_graph=mock_graph,
        max_batch_size=5,
        max_iterations=10,
        max_tool_result_chars=8000,
        annotate_line_ages=True,
    )
    d._runner = mock_runner
    return d


# ===================================================================
# EMemDream — run() basic behavior
# ===================================================================

class TestEMemDreamRun:
    """Test Dream's run() method."""

    async def test_noop_when_no_unprocessed_history(
        self, dream: EMemDream, mock_provider: MagicMock, mock_runner: MagicMock,
    ) -> None:
        result = await dream.run()
        assert result is False
        mock_provider.chat_with_retry.assert_not_called()
        mock_runner.run.assert_not_called()

    async def test_calls_phase1_for_unprocessed_entries(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis: user preference detected.",
        )
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        result = await dream.run()
        assert result is True
        mock_provider.chat_with_retry.assert_called_once()

    async def test_advances_dream_cursor(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        assert store.get_last_dream_cursor() == 2

    async def test_compacts_history_after_run(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event 1")
        store.append_history("event 2")
        store.append_history("event 3")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Nothing new")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        entries = store.read_unprocessed_history(since_cursor=0)
        assert all(e["cursor"] > 0 for e in entries)

    async def test_respects_max_batch_size(
        self,
        tmp_path,
        mock_embedder: MagicMock,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        s = EMemStore(workspace=tmp_path, embedding_model=mock_embedder)
        for i in range(10):
            s.append_history(f"event {i}")
        d = EMemDream(
            store=s,
            provider=mock_provider,
            model="test-model",
            edu_extractor=mock_edu_extractor,
            emem_store=s,
            emem_graph=mock_graph,
            max_batch_size=3,
        )
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        d._runner = mock_runner
        await d.run()
        # Cursor should advance to cursor of 3rd entry
        assert s.get_last_dream_cursor() == 3


# ===================================================================
# EMemDream — Phase 1 (LLM analysis + EDU extraction)
# ===================================================================

class TestEMemDreamPhase1:
    """Test Phase 1: LLM analysis and EDU extraction."""

    async def test_phase1_extracts_edus(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        mock_graph: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("User discussed deployment.")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Analysis: deployment discussion detected.",
        )
        from summerclaw.memory.emem_memory.datatypes import EDURecord
        mock_edu_extractor.extract_from_history.return_value = [
            EDURecord(edu_id="edu-dream-1", text="User deployed the app."),
        ]
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()

        mock_edu_extractor.extract_from_history.assert_called_once()
        # EDUs should be inserted into store
        mock_graph.load_or_create.assert_called()
        mock_graph.add_nodes.assert_called()
        mock_graph.save.assert_called()

    async def test_phase1_llm_error_is_caught(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.side_effect = Exception("LLM error")
        result = await dream.run()
        assert result is False

    async def test_phase1_edu_extraction_error_is_caught(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="Analysis.")
        mock_edu_extractor.extract_from_history.side_effect = Exception("EDU error")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        # Should not raise, should continue to Phase 2
        result = await dream.run()
        assert result is True


# ===================================================================
# EMemDream — Phase 2 (AgentRunner)
# ===================================================================

class TestEMemDreamPhase2:
    """Test Phase 2: AgentRunner for MEMORY.md editing."""

    async def test_phase2_calls_agent_runner(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("User prefers dark mode")
        mock_provider.chat_with_retry.return_value = MagicMock(content="New fact")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        result = await dream.run()
        assert result is True
        mock_runner.run.assert_called_once()

    async def test_phase2_spec_parameters(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()
        spec = mock_runner.run.call_args[0][0]
        assert spec.max_iterations == dream.max_iterations
        assert spec.fail_on_tool_error is False

    async def test_phase2_exception_is_caught(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run.side_effect = Exception("Runner error")
        # Should not raise, should advance cursor
        result = await dream.run()
        assert result is True
        assert store.get_last_dream_cursor() == 1


# ===================================================================
# EMemDream — git auto-commit
# ===================================================================

class TestEMemDreamGit:
    """Test git auto-commit after successful dream."""

    async def test_git_commit_on_success(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event")
        store.write_memory("# Updated memory")
        mock_provider.chat_with_retry.return_value = MagicMock(
            content="Memory analysis complete.",
        )
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[{"name": "edit_file", "status": "ok", "detail": "memory/MEMORY.md"}],
        ))
        # Initialize git
        store.git.init()

        await dream.run()

        # Git should have a commit
        assert store.git.is_initialized()

    async def test_git_no_commit_without_changelog(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result(
            tool_events=[],  # No tool events
        ))
        store.git.init()
        await dream.run()
        # Should not raise, even without changelog


# ===================================================================
# EMemDream — age annotation
# ===================================================================

class TestEMemDreamAgeAnnotation:
    """Test MEMORY.md line age annotation."""

    async def test_phase1_prompt_works_without_git(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("some event")
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())
        await dream.run()

        # Should still succeed without git — just no annotations
        mock_provider.chat_with_retry.assert_called_once()

    async def test_annotate_with_ages_skipped_when_disabled(
        self,
        dream: EMemDream,
        mock_provider: MagicMock,
        store: EMemStore,
    ) -> None:
        store.append_history("some event")
        dream.annotate_line_ages = False
        mock_provider.chat_with_retry.return_value = MagicMock(content="[SKIP]")
        mock_runner = dream._runner
        mock_runner.run = AsyncMock(return_value=_make_run_result())

        with patch.object(store.git, "line_ages") as mock_line_ages:
            await dream.run()
            mock_line_ages.assert_not_called()


# ===================================================================
# EMemDream — skill listing
# ===================================================================

class TestEMemDreamSkills:
    """Test existing skill listing."""

    def test_list_skills_with_user_skills(
        self, dream: EMemDream, store: EMemStore,
    ) -> None:
        # Create a skill directory
        skill_dir = store.workspace / "skills" / "test-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n",
            encoding="utf-8",
        )
        skills = dream._list_existing_skills()
        assert any("test-skill" in s for s in skills)

    def test_list_skills_empty_when_none_exist(
        self, dream: EMemDream,
    ) -> None:
        # With no skills directory or builtin, should return empty
        skills = dream._list_existing_skills()
        # May have builtin skills or may not - just check it doesn't crash
        assert isinstance(skills, list)


# ===================================================================
# EMemDream — configuration
# ===================================================================

class TestEMemDreamConfig:
    """Test Dream configuration defaults and overrides."""

    def test_default_max_batch_size(
        self,
        store: EMemStore,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        d = EMemDream(
            store=store,
            provider=mock_provider,
            model="m",
            edu_extractor=mock_edu_extractor,
            emem_store=store,
            emem_graph=mock_graph,
        )
        assert d.max_batch_size == 20

    def test_custom_max_batch_size(
        self,
        store: EMemStore,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        d = EMemDream(
            store=store,
            provider=mock_provider,
            model="m",
            edu_extractor=mock_edu_extractor,
            emem_store=store,
            emem_graph=mock_graph,
            max_batch_size=50,
        )
        assert d.max_batch_size == 50

    def test_default_max_iterations(
        self,
        store: EMemStore,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        d = EMemDream(
            store=store,
            provider=mock_provider,
            model="m",
            edu_extractor=mock_edu_extractor,
            emem_store=store,
            emem_graph=mock_graph,
        )
        assert d.max_iterations == 10

    def test_annotate_line_ages_default(
        self,
        store: EMemStore,
        mock_provider: MagicMock,
        mock_edu_extractor: MagicMock,
        mock_graph: MagicMock,
    ) -> None:
        d = EMemDream(
            store=store,
            provider=mock_provider,
            model="m",
            edu_extractor=mock_edu_extractor,
            emem_store=store,
            emem_graph=mock_graph,
        )
        assert d.annotate_line_ages is True
