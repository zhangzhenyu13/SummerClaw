"""Tests for SkillAutogen — Hermes-style background skill distillation."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.skill_autogen import SkillAutogen
from nanobot.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path)
    s.write_soul("# Soul\n- Helpful")
    s.write_user("# User\n- Developer")
    s.write_memory("# Memory\n- Project X active")
    return s


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.chat_with_retry = AsyncMock()
    return p


class TestSkillAutogenPrefix:

    def test_default_prefix_is_composite(self, store, mock_provider):
        """Default memory_algorithm_name='naive_memory' yields 'hermes--naive_memory-' prefix."""
        autogen = SkillAutogen(
            store=store,
            provider=mock_provider,
            model="test-model",
            workspace=store.workspace,
        )
        write_tool = autogen._tools.get("write_file")
        assert write_tool is not None
        # The SkillPrefixWriteFileTool should have skill_prefix="hermes--naive_memory"
        assert write_tool._skill_prefix == "hermes--naive_memory-"

    @pytest.mark.asyncio
    async def test_skill_autogen_write_prefix(self, store, mock_provider):
        """SkillAutogen write file enforces 'hermes--<algo_name>-' prefix."""
        autogen = SkillAutogen(
            store=store,
            provider=mock_provider,
            model="test-model",
            workspace=store.workspace,
            memory_algorithm_name="mastra_om_memory",
        )
        write_tool = autogen._tools.get("write_file")
        assert write_tool is not None
        assert write_tool._skill_prefix == "hermes--mastra_om_memory-"

        result = await write_tool.execute(
            path="skills/test-skill/SKILL.md",
            content="# Test skill",
        )
        assert "Successfully wrote" in result
        expected = store.workspace / "skills" / "hermes--mastra_om_memory-test-skill" / "SKILL.md"
        assert expected.exists()

    @pytest.mark.asyncio
    async def test_skill_autogen_hindsight_prefix(self, store, mock_provider):
        """SkillAutogen with hindsight_memory algorithm."""
        autogen = SkillAutogen(
            store=store,
            provider=mock_provider,
            model="test-model",
            workspace=store.workspace,
            memory_algorithm_name="hindsight_memory",
        )
        write_tool = autogen._tools.get("write_file")
        assert write_tool is not None
        assert write_tool._skill_prefix == "hermes--hindsight_memory-"

        result = await write_tool.execute(
            path="skills/skill-one/scripts/tool.py",
            content="print('hello')",
        )
        assert "Successfully wrote" in result
        expected = store.workspace / "skills" / "hermes--hindsight_memory-skill-one" / "scripts" / "tool.py"
        assert expected.exists()

    def test_explicit_memory_algorithm_name(self, store, mock_provider):
        """Passing explicit memory_algorithm_name overrides the default."""
        autogen = SkillAutogen(
            store=store,
            provider=mock_provider,
            model="test-model",
            workspace=store.workspace,
            memory_algorithm_name="supermemory_memory",
        )
        assert autogen._memory_algorithm_name == "supermemory_memory"
        write_tool = autogen._tools.get("write_file")
        assert write_tool._skill_prefix == "hermes--supermemory_memory-"


if __name__ == "__main__":
    pytest.main([__file__])
