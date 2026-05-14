"""Tests for enhanced filesystem tools: ReadFileTool, EditFileTool, ListDirTool."""

import pytest

from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    SkillPrefixWriteFileTool,
    _find_match,
)


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class TestReadFileTool:

    @pytest.fixture()
    def tool(self, tmp_path):
        return ReadFileTool(workspace=tmp_path)

    @pytest.fixture()
    def sample_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")
        return f

    @pytest.mark.asyncio
    async def test_basic_read_has_line_numbers(self, tool, sample_file):
        result = await tool.execute(path=str(sample_file))
        assert "1| line 1" in result
        assert "20| line 20" in result

    @pytest.mark.asyncio
    async def test_offset_and_limit(self, tool, sample_file):
        result = await tool.execute(path=str(sample_file), offset=5, limit=3)
        assert "5| line 5" in result
        assert "7| line 7" in result
        assert "8| line 8" not in result
        assert "Use offset=8 to continue" in result

    @pytest.mark.asyncio
    async def test_offset_beyond_end(self, tool, sample_file):
        result = await tool.execute(path=str(sample_file), offset=999)
        assert "Error" in result
        assert "beyond end" in result

    @pytest.mark.asyncio
    async def test_end_of_file_marker(self, tool, sample_file):
        result = await tool.execute(path=str(sample_file), offset=1, limit=9999)
        assert "End of file" in result

    @pytest.mark.asyncio
    async def test_empty_file(self, tool, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = await tool.execute(path=str(f))
        assert "Empty file" in result

    @pytest.mark.asyncio
    async def test_image_file_returns_multimodal_blocks(self, tool, tmp_path):
        f = tmp_path / "pixel.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-data")

        result = await tool.execute(path=str(f))

        assert isinstance(result, list)
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result[0]["_meta"]["path"] == str(f)
        assert result[1] == {"type": "text", "text": f"(Image file: {f})"}

    @pytest.mark.asyncio
    async def test_file_not_found(self, tool, tmp_path):
        result = await tool.execute(path=str(tmp_path / "nope.txt"))
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_missing_path_returns_clear_error(self, tool):
        result = await tool.execute()
        assert result == "Error reading file: Unknown path"

    @pytest.mark.asyncio
    async def test_char_budget_trims(self, tool, tmp_path):
        """When the selected slice exceeds _MAX_CHARS the output is trimmed."""
        f = tmp_path / "big.txt"
        # Each line is ~110 chars, 2000 lines ≈ 220 KB > 128 KB limit
        f.write_text("\n".join("x" * 110 for _ in range(2000)), encoding="utf-8")
        result = await tool.execute(path=str(f))
        assert len(result) <= ReadFileTool._MAX_CHARS + 500  # small margin for footer
        assert "Use offset=" in result


# ---------------------------------------------------------------------------
# _find_match  (unit tests for the helper)
# ---------------------------------------------------------------------------

class TestFindMatch:

    def test_exact_match(self):
        match, count = _find_match("hello world", "world")
        assert match == "world"
        assert count == 1

    def test_exact_no_match(self):
        match, count = _find_match("hello world", "xyz")
        assert match is None
        assert count == 0

    def test_crlf_normalisation(self):
        # Caller normalises CRLF before calling _find_match, so test with
        # pre-normalised content to verify exact match still works.
        content = "line1\nline2\nline3"
        old_text = "line1\nline2\nline3"
        match, count = _find_match(content, old_text)
        assert match is not None
        assert count == 1

    def test_line_trim_fallback(self):
        content = "    def foo():\n        pass\n"
        old_text = "def foo():\n    pass"
        match, count = _find_match(content, old_text)
        assert match is not None
        assert count == 1
        # The returned match should be the *original* indented text
        assert "    def foo():" in match

    def test_line_trim_multiple_candidates(self):
        content = "  a\n  b\n  a\n  b\n"
        old_text = "a\nb"
        match, count = _find_match(content, old_text)
        assert count == 2

    def test_empty_old_text(self):
        match, count = _find_match("hello", "")
        # Empty string is always "in" any string via exact match
        assert match == ""


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------

class TestEditFileTool:

    @pytest.fixture()
    def tool(self, tmp_path):
        return EditFileTool(workspace=tmp_path)

    @pytest.mark.asyncio
    async def test_exact_match(self, tool, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("hello world", encoding="utf-8")
        result = await tool.execute(path=str(f), old_text="world", new_text="earth")
        assert "Successfully" in result
        assert f.read_text() == "hello earth"

    @pytest.mark.asyncio
    async def test_crlf_normalisation(self, tool, tmp_path):
        f = tmp_path / "crlf.py"
        f.write_bytes(b"line1\r\nline2\r\nline3")
        result = await tool.execute(
            path=str(f), old_text="line1\nline2", new_text="LINE1\nLINE2",
        )
        assert "Successfully" in result
        raw = f.read_bytes()
        assert b"LINE1" in raw
        # CRLF line endings should be preserved throughout the file
        assert b"\r\n" in raw

    @pytest.mark.asyncio
    async def test_trim_fallback(self, tool, tmp_path):
        f = tmp_path / "indent.py"
        f.write_text("    def foo():\n        pass\n", encoding="utf-8")
        result = await tool.execute(
            path=str(f), old_text="def foo():\n    pass", new_text="def bar():\n    return 1",
        )
        assert "Successfully" in result
        assert "bar" in f.read_text()

    @pytest.mark.asyncio
    async def test_ambiguous_match(self, tool, tmp_path):
        f = tmp_path / "dup.py"
        f.write_text("aaa\nbbb\naaa\nbbb\n", encoding="utf-8")
        result = await tool.execute(path=str(f), old_text="aaa\nbbb", new_text="xxx")
        assert "appears" in result.lower() or "Warning" in result

    @pytest.mark.asyncio
    async def test_replace_all(self, tool, tmp_path):
        f = tmp_path / "multi.py"
        f.write_text("foo bar foo bar foo", encoding="utf-8")
        result = await tool.execute(
            path=str(f), old_text="foo", new_text="baz", replace_all=True,
        )
        assert "Successfully" in result
        assert f.read_text() == "baz bar baz bar baz"

    @pytest.mark.asyncio
    async def test_not_found(self, tool, tmp_path):
        f = tmp_path / "nf.py"
        f.write_text("hello", encoding="utf-8")
        result = await tool.execute(path=str(f), old_text="xyz", new_text="abc")
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_missing_new_text_returns_clear_error(self, tool, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("hello", encoding="utf-8")
        result = await tool.execute(path=str(f), old_text="hello")
        assert result == "Error editing file: Unknown new_text"


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------

class TestListDirTool:

    @pytest.fixture()
    def tool(self, tmp_path):
        return ListDirTool(workspace=tmp_path)

    @pytest.fixture()
    def populated_dir(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("pass")
        (tmp_path / "src" / "utils.py").write_text("pass")
        (tmp_path / "README.md").write_text("hi")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg").mkdir()
        return tmp_path

    @pytest.mark.asyncio
    async def test_basic_list(self, tool, populated_dir):
        result = await tool.execute(path=str(populated_dir))
        assert "README.md" in result
        assert "src" in result
        # .git and node_modules should be ignored
        assert ".git" not in result
        assert "node_modules" not in result

    @pytest.mark.asyncio
    async def test_recursive(self, tool, populated_dir):
        result = await tool.execute(path=str(populated_dir), recursive=True)
        # Normalize path separators for cross-platform compatibility
        normalized = result.replace("\\", "/")
        assert "src/main.py" in normalized
        assert "src/utils.py" in normalized
        assert "README.md" in result
        # Ignored dirs should not appear
        assert ".git" not in result
        assert "node_modules" not in result

    @pytest.mark.asyncio
    async def test_max_entries_truncation(self, tool, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.txt").write_text("x")
        result = await tool.execute(path=str(tmp_path), max_entries=3)
        assert "truncated" in result
        assert "3 of 10" in result

    @pytest.mark.asyncio
    async def test_empty_dir(self, tool, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        result = await tool.execute(path=str(d))
        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_not_found(self, tool, tmp_path):
        result = await tool.execute(path=str(tmp_path / "nope"))
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_missing_path_returns_clear_error(self, tool):
        result = await tool.execute()
        assert result == "Error listing directory: Unknown path"


# ---------------------------------------------------------------------------
# Workspace restriction + extra_allowed_dirs
# ---------------------------------------------------------------------------

class TestWorkspaceRestriction:

    @pytest.mark.asyncio
    async def test_read_blocked_outside_workspace(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("top secret")

        tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
        result = await tool.execute(path=str(secret))
        assert "Error" in result
        assert "outside" in result.lower()

    @pytest.mark.asyncio
    async def test_read_allowed_with_extra_dir(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_file = skills_dir / "test_skill" / "SKILL.md"
        skill_file.parent.mkdir()
        skill_file.write_text("# Test Skill\nDo something.")

        tool = ReadFileTool(
            workspace=workspace, allowed_dir=workspace,
            extra_allowed_dirs=[skills_dir],
        )
        result = await tool.execute(path=str(skill_file))
        assert "Test Skill" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_read_allowed_in_media_dir(self, tmp_path, monkeypatch):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        media_dir = tmp_path / "media"
        media_dir.mkdir()
        media_file = media_dir / "photo.txt"
        media_file.write_text("shared media", encoding="utf-8")

        monkeypatch.setattr("nanobot.agent.tools.filesystem.get_media_dir", lambda: media_dir)

        tool = ReadFileTool(workspace=workspace, allowed_dir=workspace)
        result = await tool.execute(path=str(media_file))
        assert "shared media" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_extra_dirs_does_not_widen_write(self, tmp_path):
        from nanobot.agent.tools.filesystem import WriteFileTool

        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        tool = WriteFileTool(workspace=workspace, allowed_dir=workspace)
        result = await tool.execute(path=str(outside / "hack.txt"), content="pwned")
        assert "Error" in result
        assert "outside" in result.lower()

    @pytest.mark.asyncio
    async def test_read_still_blocked_for_unrelated_dir(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        unrelated = tmp_path / "other"
        unrelated.mkdir()
        secret = unrelated / "secret.txt"
        secret.write_text("nope")

        tool = ReadFileTool(
            workspace=workspace, allowed_dir=workspace,
            extra_allowed_dirs=[skills_dir],
        )
        result = await tool.execute(path=str(secret))
        assert "Error" in result
        assert "outside" in result.lower()

    @pytest.mark.asyncio
    async def test_workspace_file_still_readable_with_extra_dirs(self, tmp_path):
        """Adding extra_allowed_dirs must not break normal workspace reads."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        ws_file = workspace / "README.md"
        ws_file.write_text("hello from workspace")
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        tool = ReadFileTool(
            workspace=workspace, allowed_dir=workspace,
            extra_allowed_dirs=[skills_dir],
        )
        result = await tool.execute(path=str(ws_file))
        assert "hello from workspace" in result
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_edit_blocked_in_extra_dir(self, tmp_path):
        """edit_file must not be able to modify files in extra_allowed_dirs."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        skill_file = skills_dir / "weather" / "SKILL.md"
        skill_file.parent.mkdir()
        skill_file.write_text("# Weather\nOriginal content.")

        tool = EditFileTool(workspace=workspace, allowed_dir=workspace)
        result = await tool.execute(
            path=str(skill_file),
            old_text="Original content.",
            new_text="Hacked content.",
        )
        assert "Error" in result
        assert "outside" in result.lower()
        assert skill_file.read_text() == "# Weather\nOriginal content."


# ---------------------------------------------------------------------------
# SkillPrefixWriteFileTool
# ---------------------------------------------------------------------------

class TestSkillPrefixWriteFileTool:

    # -- helpers ----------------------------------------------------------

    @pytest.fixture()
    def skills_dir(self, tmp_path):
        d = tmp_path / "skills"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- basic prefix (legacy style: "dreamed-") --------------------------

    @pytest.mark.asyncio
    async def test_basic_prefix(self, tmp_path, skills_dir):
        """Simple prefix like 'dreamed-' still works."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="dreamed",
            workspace=tmp_path,
            allowed_dir=skills_dir,
        )
        result = await tool.execute(
            path="skills/test-skill/SKILL.md",
            content="# Test",
        )
        assert "Successfully wrote" in result
        assert (skills_dir / "dreamed-test-skill" / "SKILL.md").exists()

    # -- composite prefix with double-dash (new style: "dreamed--<algo>-") -

    @pytest.mark.asyncio
    async def test_composite_prefix_with_double_dash(self, tmp_path, skills_dir):
        """Composite prefix 'dreamed--naive_memory-' correctly rewrites path."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="dreamed--naive_memory",
            workspace=tmp_path,
            allowed_dir=skills_dir,
        )
        result = await tool.execute(
            path="skills/test-skill/SKILL.md",
            content="# Test",
        )
        assert "Successfully wrote" in result
        assert (skills_dir / "dreamed--naive_memory-test-skill" / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_composite_prefix_hermes_style(self, tmp_path, skills_dir):
        """Composite prefix 'hermes--mastra_om_memory-' works as well."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="hermes--mastra_om_memory",
            workspace=tmp_path,
            allowed_dir=skills_dir,
        )
        result = await tool.execute(
            path="skills/my-skill/SKILL.md",
            content="# Test",
        )
        assert "Successfully wrote" in result
        assert (skills_dir / "hermes--mastra_om_memory-my-skill" / "SKILL.md").exists()

    # -- already has prefix (idempotent) -----------------------------------

    @pytest.mark.asyncio
    async def test_prefix_already_present(self, tmp_path, skills_dir):
        """If the directory name already starts with the prefix, do not double-prefix."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="dreamed",
            workspace=tmp_path,
            allowed_dir=skills_dir,
        )
        result = await tool.execute(
            path="skills/dreamed-test-skill/SKILL.md",
            content="# Test",
        )
        assert "Successfully wrote" in result
        assert (skills_dir / "dreamed-test-skill" / "SKILL.md").exists()

    @pytest.mark.asyncio
    async def test_composite_prefix_already_present(self, tmp_path, skills_dir):
        """Composite prefix already present - no double prefixing."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="dreamed--naive_memory",
            workspace=tmp_path,
            allowed_dir=skills_dir,
        )
        result = await tool.execute(
            path="skills/dreamed--naive_memory-test-skill/SKILL.md",
            content="# Test",
        )
        assert "Successfully wrote" in result
        assert (skills_dir / "dreamed--naive_memory-test-skill" / "SKILL.md").exists()

    # -- prefix with scripts sub-path -------------------------------------

    @pytest.mark.asyncio
    async def test_composite_prefix_with_scripts_subpath(self, tmp_path, skills_dir):
        """Composite prefix also works for scripts/ subdirectory."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="dreamed--hindsight_memory",
            workspace=tmp_path,
            allowed_dir=skills_dir,
        )
        result = await tool.execute(
            path="skills/my-skill/scripts/tool.py",
            content="print('hello')",
        )
        assert "Successfully wrote" in result
        assert (skills_dir / "dreamed--hindsight_memory-my-skill" / "scripts" / "tool.py").exists()

    # -- non-skills path is NOT rewritten ---------------------------------

    @pytest.mark.asyncio
    async def test_non_skills_path_not_rewritten(self, tmp_path, skills_dir):
        """Prefix enforcement only applies under skills/."""
        tool = SkillPrefixWriteFileTool(
            skill_prefix="dreamed--naive_memory",
            workspace=tmp_path,
            allowed_dir=tmp_path,  # allow writing anywhere
        )
        result = await tool.execute(
            path="README.md",
            content="# Test",
        )
        assert "Successfully wrote" in result
        assert (tmp_path / "README.md").exists()
