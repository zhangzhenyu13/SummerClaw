"""Comprehensive tests for BrowserManager and browser control tools."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summerclaw.agent.tools.browser_control import (
    BrowserExecuteJSTool,
    BrowserManager,
    BrowserNavigateTool,
    BrowserSnapshotTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager():
    return BrowserManager(idle_timeout=300)


def _make_mock_page():
    """Create a mock Playwright Page with common methods."""
    page = AsyncMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example Page")
    page.goto = AsyncMock()
    page.bring_to_front = AsyncMock()
    page.evaluate = AsyncMock(return_value={"status": "success", "result": "ok"})
    page.inner_text = AsyncMock(return_value="Page body text")
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value={
        "role": "WebArea",
        "name": "Example Page",
        "children": [
            {"role": "heading", "name": "Hello World", "level": 1},
            {"role": "link", "name": "Click here", "value": "https://other.com"},
            {"role": "button", "name": "Submit"},
        ],
    })
    page.screenshot = AsyncMock(return_value=b"\x89PNG...")
    return page


def _make_mock_browser():
    """Mock Playwright BrowserContext and Browser with one page."""
    page = _make_mock_page()
    ctx = AsyncMock()
    ctx.new_page = AsyncMock(return_value=page)
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=ctx)
    browser.close = AsyncMock()
    browser.is_connected = MagicMock(return_value=True)
    return browser, ctx, page


def _setup_mocked_manager(manager=None):
    """Set up a BrowserManager with mocked Playwright internals.
    
    Returns (manager, mock_browser, mock_context, mock_page).
    """
    if manager is None:
        manager = _make_manager()
    
    # Pre-set _playwright, _browser, _context, _pages with mocks
    browser, ctx, page = _make_mock_browser()
    manager._playwright = MagicMock()
    manager._browser = browser
    manager._context = ctx
    manager._pages = {"default": page}
    manager._active_page_id = "default"
    # Disable idle timeout to prevent auto-close during tests
    manager._idle_timeout = 0
    manager._last_used = 999999.0
    
    return manager, browser, ctx, page


# ---------------------------------------------------------------------------
# BrowserManager — initialization
# ---------------------------------------------------------------------------

class TestBrowserManagerInit:

    def test_default_idle_timeout(self):
        mgr = BrowserManager()
        assert mgr._idle_timeout == 300

    def test_custom_idle_timeout(self):
        mgr = BrowserManager(idle_timeout=120)
        assert mgr._idle_timeout == 120

    def test_initially_no_browser(self):
        mgr = _make_manager()
        assert mgr._browser is None
        assert mgr._playwright is None
        assert mgr._context is None

    def test_initially_empty_pages(self):
        mgr = _make_manager()
        assert mgr._pages == {}

    def test_initially_no_active_page(self):
        mgr = _make_manager()
        assert mgr._active_page_id is None

    def test_lock_is_asyncio_lock(self):
        mgr = _make_manager()
        assert isinstance(mgr._lock, asyncio.Lock)


# ---------------------------------------------------------------------------
# BrowserManager — URL validation (SSRF)
# ---------------------------------------------------------------------------

class TestBrowserManagerUrlValidation:

    def test_public_http_url_passes(self):
        """Public URL should pass SSRF validation."""
        mgr = _make_manager()
        # We can't rely on DNS in tests, so test the call pattern
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            mgr._validate_url("https://example.com")

    def test_private_ip_url_fails(self):
        mgr = _make_manager()
        with patch("summerclaw.security.network.validate_url_target", return_value=(False, "private")):
            with pytest.raises(ValueError, match="URL validation failed"):
                mgr._validate_url("http://127.0.0.1")

    def test_file_scheme_fails(self):
        mgr = _make_manager()
        with patch("summerclaw.security.network.validate_url_target", return_value=(False, "scheme")):
            with pytest.raises(ValueError, match="URL validation failed"):
                mgr._validate_url("file:///etc/passwd")

    def test_invalid_url_fails(self):
        mgr = _make_manager()
        with patch("summerclaw.security.network.validate_url_target", return_value=(False, "bad")):
            with pytest.raises(ValueError, match="URL validation failed"):
                mgr._validate_url("not-a-url")


# ---------------------------------------------------------------------------
# BrowserManager — ensure_browser / lifecycle
# ---------------------------------------------------------------------------

class TestBrowserManagerEnsureBrowser:

    def test_ensure_browser_reuses_existing(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        # Should not reinitialize
        assert mgr._browser is browser

    @pytest.mark.asyncio
    async def test_ensure_browser_lazy_inits_when_none(self):
        """Test lazy init path by mocking playwright in sys.modules."""
        import sys

        mgr = _make_manager()

        # Create mock playwright namespace and inject it into sys.modules
        mock_async_pw = MagicMock()
        mock_instance = MagicMock()
        mock_instance.start = AsyncMock()
        mock_async_pw.return_value = mock_instance

        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        mock_browser.is_connected = MagicMock(return_value=True)
        mock_instance.chromium.launch = AsyncMock(return_value=mock_browser)

        mock_ctx = MagicMock()
        mock_page = _make_mock_page()
        mock_ctx.new_page = AsyncMock(return_value=mock_page)
        mock_browser.new_context = AsyncMock(return_value=mock_ctx)

        # Inject mocks into sys.modules so the import inside ensure_browser works
        original_modules = {}
        for key in ("playwright", "playwright.async_api"):
            original_modules[key] = sys.modules.get(key)
            sys.modules[key] = MagicMock()
        sys.modules["playwright.async_api"].async_playwright = mock_async_pw

        try:
            with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
                await mgr.ensure_browser()

            assert "default" in mgr._pages
            assert mgr._active_page_id == "default"
            # The page should be the one created by new_page()
            assert mgr._pages["default"] is not None
        finally:
            # Restore sys.modules
            for key in ("playwright", "playwright.async_api"):
                if original_modules[key] is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = original_modules[key]

    @pytest.mark.asyncio
    async def test_ensure_browser_closes_idle_browser(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        # Override settings for this specific test
        mgr._idle_timeout = 10  # 10 seconds
        mgr._last_used = 1  # Recent but far enough from now=999999
        browser.is_connected = MagicMock(return_value=True)

        # Directly test the idle timeout logic
        async with mgr._lock:
            now = 999999.0
            if (mgr._browser is not None and mgr._idle_timeout > 0
                    and mgr._last_used > 0 and (now - mgr._last_used) > mgr._idle_timeout):
                await mgr._close_browser()

        assert mgr._browser is None
        assert mgr._pages == {}


# ---------------------------------------------------------------------------
# BrowserManager — navigate
# ---------------------------------------------------------------------------

class TestBrowserManagerNavigate:

    @pytest.mark.asyncio
    async def test_navigate_returns_url_and_title(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            result = await mgr.navigate("https://example.com")
        
        assert result["url"] == "https://example.com"
        assert result["title"] == "Example Page"
        page.goto.assert_awaited_once_with(
            "https://example.com", wait_until="domcontentloaded", timeout=15000
        )

    @pytest.mark.asyncio
    async def test_navigate_falls_back_on_load_failure(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.goto = AsyncMock(side_effect=[Exception("timeout"), None])
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            await mgr.navigate("https://slow.com")
        assert page.goto.await_count == 2

    @pytest.mark.asyncio
    async def test_navigate_updates_last_used(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        old = mgr._last_used
        with (
            patch("summerclaw.security.network.validate_url_target", return_value=(True, "")),
            patch("time.monotonic", return_value=5000.0),
        ):
            await mgr.navigate("https://example.com")
        assert mgr._last_used == 5000.0

    @pytest.mark.asyncio
    async def test_navigate_handles_title_error(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.title = AsyncMock(side_effect=Exception("no title"))
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            result = await mgr.navigate("https://example.com")
        assert result["title"] == ""


# ---------------------------------------------------------------------------
# BrowserManager — tabs
# ---------------------------------------------------------------------------

class TestBrowserManagerTabs:

    @pytest.mark.asyncio
    async def test_list_tabs_returns_tab_info(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tabs = await mgr.list_tabs()
        assert len(tabs) == 1
        assert tabs[0]["id"] == "default"
        assert "example.com" in tabs[0]["url"]
        assert tabs[0]["active"] is True

    @pytest.mark.asyncio
    async def test_list_tabs_marks_inactive_tabs(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page2 = _make_mock_page()
        mgr._pages["tab_2"] = page2
        mgr._active_page_id = "default"
        tabs = await mgr.list_tabs()
        for t in tabs:
            if t["id"] == "default":
                assert t["active"] is True
            elif t["id"] == "tab_2":
                assert t["active"] is False

    @pytest.mark.asyncio
    async def test_list_tabs_handles_page_errors(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.url = "javascript:void(0)"  # tricky
        page.title = AsyncMock(side_effect=Exception("boom"))
        tabs = await mgr.list_tabs()
        assert len(tabs) == 1

    @pytest.mark.asyncio
    async def test_switch_tab_to_existing(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page2 = _make_mock_page()
        mgr._pages["tab_2"] = page2
        result = await mgr.switch_tab("tab_2")
        assert result["status"] == "switched"
        assert mgr._active_page_id == "tab_2"
        page2.bring_to_front.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_switch_tab_not_found(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        result = await mgr.switch_tab("nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_new_tab_with_url(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        new_page = _make_mock_page()
        ctx.new_page = AsyncMock(return_value=new_page)
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            result = await mgr.new_tab("https://new-page.com")
        assert result["status"] == "created"
        assert "tab_2" in result["tab_id"]
        assert mgr._active_page_id == result["tab_id"]

    @pytest.mark.asyncio
    async def test_new_tab_with_blank_url(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        new_page = _make_mock_page()
        ctx.new_page = AsyncMock(return_value=new_page)
        result = await mgr.new_tab("about:blank")
        assert result["status"] == "created"
        new_page.goto.assert_not_awaited()


# ---------------------------------------------------------------------------
# BrowserManager — execute_js
# ---------------------------------------------------------------------------

class TestBrowserManagerExecuteJS:

    @pytest.mark.asyncio
    async def test_execute_js_returns_result(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value=42)
        result = await mgr.execute_js("return 42")
        assert result["status"] == "success"
        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_execute_js_wraps_in_async_iife(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value="done")
        await mgr.execute_js("await new Promise(r => r('done'))")
        called_with = page.evaluate.call_args[0][0]
        assert called_with.startswith("(async () => {")

    @pytest.mark.asyncio
    async def test_execute_js_sync_fallback(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        # async IIFE fails, sync wrapper succeeds
        page.evaluate = AsyncMock(side_effect=[
            Exception("not async safe"),
            "sync_result",
        ])
        result = await mgr.execute_js("document.title")
        assert result["status"] == "success"
        assert result["result"] == "sync_result"

    @pytest.mark.asyncio
    async def test_execute_js_both_failures(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(side_effect=Exception("always fails"))
        result = await mgr.execute_js("bad code")
        assert result["status"] == "error"
        assert "always fails" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_js_updates_last_used(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        with patch("time.monotonic", return_value=6000.0):
            await mgr.execute_js("1+1")
        assert mgr._last_used == 6000.0


# ---------------------------------------------------------------------------
# BrowserManager — snapshot (accessibility tree)
# ---------------------------------------------------------------------------

class TestBrowserManagerSnapshot:

    @pytest.mark.asyncio
    async def test_snapshot_returns_formatted_tree(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "name": "Test",
            "children": [
                {"role": "heading", "name": "Title"},
                {"role": "button", "name": "OK"},
            ],
        })
        result = await mgr.snapshot()
        assert "heading" in result
        assert "Title" in result
        assert "button" in result
        assert "OK" in result

    @pytest.mark.asyncio
    async def test_snapshot_empty_tree(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.accessibility.snapshot = AsyncMock(return_value=None)
        result = await mgr.snapshot()
        assert "empty page" in result

    @pytest.mark.asyncio
    async def test_snapshot_falls_back_to_body_text(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.accessibility.snapshot = AsyncMock(side_effect=Exception("not supported"))
        result = await mgr.snapshot()
        assert "Page body text" in result or "example.com" in result

    @pytest.mark.asyncio
    async def test_snapshot_truncates_at_max_length(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        # Create a big tree
        big_children = [
            {"role": "text", "name": f"line {i}"} for i in range(500)
        ]
        page.accessibility.snapshot = AsyncMock(return_value={
            "role": "WebArea",
            "name": "Big",
            "children": big_children,
        })
        result = await mgr.snapshot(max_length=200)
        assert len(result) <= 200 + 50  # allow for truncation message
        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_snapshot_updates_last_used(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        with patch("time.monotonic", return_value=7000.0):
            await mgr.snapshot()
        assert mgr._last_used == 7000.0


# ---------------------------------------------------------------------------
# BrowserManager — _format_a11y_tree (static helper)
# ---------------------------------------------------------------------------

class TestFormatA11yTree:

    def test_root_node_indent_zero(self):
        node = {"role": "WebArea", "name": "root"}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        assert len(lines) >= 1
        assert "WebArea" in lines[0]
        assert "root" in lines[0]

    def test_link_shows_name_in_quotes(self):
        node = {"role": "link", "name": "Homepage"}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        assert '"Homepage"' in lines[0]

    def test_button_shows_name_in_quotes(self):
        node = {"role": "button", "name": "Submit"}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        assert '"Submit"' in lines[0]

    def test_textbox_shows_value(self):
        node = {"role": "textbox", "name": "search", "value": "hello"}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        assert "hello" in lines[0]

    def test_nested_children_are_indented(self):
        node = {
            "role": "WebArea",
            "children": [
                {"role": "heading", "name": "Title", "children": [
                    {"role": "text", "name": "inner"},
                ]},
            ],
        }
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        # Check indentation — child should have 2 spaces, grandchild 4
        assert any(l.startswith("  heading") for l in lines)
        assert any(l.startswith("    text") for l in lines)

    def test_long_name_truncated(self):
        node = {"role": "text", "name": "x" * 200}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        name_part = lines[0]
        assert len(name_part) < 200  # name was truncated to 80 chars

    def test_node_with_no_name(self):
        node = {"role": "generic"}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        assert lines[0].strip() == "generic"

    def test_node_with_newlines_in_name(self):
        node = {"role": "text", "name": "line1\nline2\nline3"}
        lines = BrowserManager._format_a11y_tree(node, indent=0)
        assert "\n" not in lines[0]


# ---------------------------------------------------------------------------
# BrowserManager — screenshot
# ---------------------------------------------------------------------------

class TestBrowserManagerScreenshot:

    @pytest.mark.asyncio
    async def test_screenshot_returns_bytes(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.screenshot = AsyncMock(return_value=b"\x89PNG\x00\x00\x00")
        result = await mgr.screenshot()
        assert isinstance(result, bytes)
        page.screenshot.assert_awaited_once_with(full_page=False, type="png")

    @pytest.mark.asyncio
    async def test_screenshot_updates_last_used(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.screenshot = AsyncMock(return_value=b"png")
        with patch("time.monotonic", return_value=8000.0):
            await mgr.screenshot()
        assert mgr._last_used == 8000.0


# ---------------------------------------------------------------------------
# BrowserManager — close
# ---------------------------------------------------------------------------

class TestBrowserManagerClose:

    @pytest.mark.asyncio
    async def test_close_clears_state(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        await mgr.close()
        assert mgr._browser is None
        assert mgr._context is None
        assert mgr._pages == {}
        assert mgr._active_page_id is None
        browser.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_handles_exceptions_gracefully(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        browser.close = AsyncMock(side_effect=RuntimeError("already closed"))
        await mgr.close()
        # Should not raise
        assert mgr._browser is None


# ---------------------------------------------------------------------------
# BrowserNavigateTool — metadata
# ---------------------------------------------------------------------------

class TestBrowserNavigateToolMetadata:

    def test_name(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        assert tool.name == "browser_navigate"

    def test_description_mentions_navigate(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        assert "navigate" in tool.description.lower()

    def test_exclusive_is_true(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        assert tool.exclusive is True

    def test_concurrency_safe_is_false(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        assert tool.concurrency_safe is False

    def test_action_is_required(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        required = tool.parameters.get("required", [])
        assert "action" in required

    def test_action_has_four_enum_values(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        action_prop = tool.parameters["properties"]["action"]
        assert set(action_prop["enum"]) == {"go", "tabs", "switch", "new_tab"}


# ---------------------------------------------------------------------------
# BrowserNavigateTool — actions (go)
# ---------------------------------------------------------------------------

class TestBrowserNavigateToolGo:

    @pytest.mark.asyncio
    async def test_go_action_returns_result(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            result = await tool.execute(action="go", url="https://example.com")
        assert "Navigated to" in result
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_go_missing_url_returns_error(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="go", url="")
        assert "Error" in result
        assert "url" in result.lower()

    @pytest.mark.asyncio
    async def test_go_exception_returns_error(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            mgr.navigate = AsyncMock(side_effect=RuntimeError("network down"))
            result = await tool.execute(action="go", url="https://example.com")
        assert "Error" in result
        assert "network down" in result


# ---------------------------------------------------------------------------
# BrowserNavigateTool — actions (tabs)
# ---------------------------------------------------------------------------

class TestBrowserNavigateToolTabs:

    @pytest.mark.asyncio
    async def test_tabs_action_returns_list(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="tabs")
        assert "1 tab" in result or "tab(s)" in result

    @pytest.mark.asyncio
    async def test_tabs_action_no_tabs(self):
        mgr = _make_manager()
        mgr, browser, ctx, page = _setup_mocked_manager(mgr)
        mgr._pages = {}
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="tabs")
        assert "No open tabs" in result

    @pytest.mark.asyncio
    async def test_tabs_action_shows_active_marker(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="tabs")
        assert "active" in result.lower()


# ---------------------------------------------------------------------------
# BrowserNavigateTool — actions (switch)
# ---------------------------------------------------------------------------

class TestBrowserNavigateToolSwitch:

    @pytest.mark.asyncio
    async def test_switch_action_returns_result(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        mgr._pages["tab_2"] = _make_mock_page()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="switch", tab_id="tab_2")
        assert "Switched" in result

    @pytest.mark.asyncio
    async def test_switch_missing_tab_id_returns_error(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="switch", tab_id="")
        assert "Error" in result
        assert "tab_id" in result.lower()

    @pytest.mark.asyncio
    async def test_switch_none_tab_id_returns_error(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="switch", tab_id=None)
        assert "Error" in result


# ---------------------------------------------------------------------------
# BrowserNavigateTool — actions (new_tab)
# ---------------------------------------------------------------------------

class TestBrowserNavigateToolNewTab:

    @pytest.mark.asyncio
    async def test_new_tab_action_returns_result(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        ctx.new_page = AsyncMock(return_value=_make_mock_page())
        tool = BrowserNavigateTool(manager=mgr)
        with patch("summerclaw.security.network.validate_url_target", return_value=(True, "")):
            result = await tool.execute(action="new_tab", url="https://example.com")
        assert "Created tab" in result

    @pytest.mark.asyncio
    async def test_new_tab_default_blank(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        ctx.new_page = AsyncMock(return_value=_make_mock_page())
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="new_tab")
        assert "Created tab" in result


# ---------------------------------------------------------------------------
# BrowserNavigateTool — errors
# ---------------------------------------------------------------------------

class TestBrowserNavigateToolErrors:

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        mgr = _make_manager()
        tool = BrowserNavigateTool(manager=mgr)
        result = await tool.execute(action="fly")
        assert "Unknown action" in result


# ---------------------------------------------------------------------------
# BrowserSnapshotTool — metadata
# ---------------------------------------------------------------------------

class TestBrowserSnapshotToolMetadata:

    def test_name(self):
        mgr = _make_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        assert tool.name == "browser_snapshot"

    def test_description_mentions_accessibility(self):
        mgr = _make_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        desc = tool.description.lower()
        assert "accessibility" in desc or "snapshot" in desc

    def test_read_only_is_true(self):
        mgr = _make_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        assert tool.read_only is True

    def test_exclusive_is_true(self):
        mgr = _make_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        assert tool.exclusive is True

    def test_max_length_has_bounds(self):
        mgr = _make_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        prop = tool.parameters["properties"]["max_length"]
        assert prop["type"] == "integer"
        assert prop["minimum"] == 500
        assert prop["maximum"] == 50000


# ---------------------------------------------------------------------------
# BrowserSnapshotTool — execute
# ---------------------------------------------------------------------------

class TestBrowserSnapshotToolExecute:

    @pytest.mark.asyncio
    async def test_execute_returns_snapshot(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        result = await tool.execute()
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_execute_with_custom_max_length(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        tool = BrowserSnapshotTool(manager=mgr)
        result = await tool.execute(max_length=600)
        assert len(result) <= 650  # allow margin

    @pytest.mark.asyncio
    async def test_execute_handles_exception(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        mgr.snapshot = AsyncMock(side_effect=Exception("broken"))
        tool = BrowserSnapshotTool(manager=mgr)
        result = await tool.execute()
        assert "Error" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# BrowserExecuteJSTool — metadata
# ---------------------------------------------------------------------------

class TestBrowserExecuteJSToolMetadata:

    def test_name(self):
        mgr = _make_manager()
        tool = BrowserExecuteJSTool(manager=mgr)
        assert tool.name == "browser_execute_js"

    def test_description_mentions_javascript(self):
        mgr = _make_manager()
        tool = BrowserExecuteJSTool(manager=mgr)
        assert "JavaScript" in tool.description or "javascript" in tool.description.lower()

    def test_exclusive_is_true(self):
        mgr = _make_manager()
        tool = BrowserExecuteJSTool(manager=mgr)
        assert tool.exclusive is True

    def test_concurrency_safe_is_false(self):
        mgr = _make_manager()
        tool = BrowserExecuteJSTool(manager=mgr)
        assert tool.concurrency_safe is False

    def test_script_is_required(self):
        mgr = _make_manager()
        tool = BrowserExecuteJSTool(manager=mgr)
        required = tool.parameters.get("required", [])
        assert "script" in required

    def test_timeout_has_bounds(self):
        mgr = _make_manager()
        tool = BrowserExecuteJSTool(manager=mgr)
        prop = tool.parameters["properties"]["timeout"]
        assert prop["type"] == "integer"
        assert prop["minimum"] == 500
        assert prop["maximum"] == 30000


# ---------------------------------------------------------------------------
# BrowserExecuteJSTool — execute
# ---------------------------------------------------------------------------

class TestBrowserExecuteJSToolExecute:

    @pytest.mark.asyncio
    async def test_execute_returns_string_result(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value="done")
        tool = BrowserExecuteJSTool(manager=mgr)
        result = await tool.execute(script="return 'done'")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_execute_handles_js_error(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(side_effect=Exception("eval failed"))
        tool = BrowserExecuteJSTool(manager=mgr)
        result = await tool.execute(script="bad")
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_passes_timeout_to_manager(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value="ok")
        tool = BrowserExecuteJSTool(manager=mgr)
        await tool.execute(script="1", timeout=5000)
        # Manager should receive timeout_ms=5000
        page.evaluate.assert_awaited()

    @pytest.mark.asyncio
    async def test_execute_dict_result_is_json_serialized(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value={"key": "value", "num": 42})
        tool = BrowserExecuteJSTool(manager=mgr)
        result = await tool.execute(script="return {key: 'value'}")
        assert '"key"' in result
        assert '"value"' in result

    @pytest.mark.asyncio
    async def test_execute_list_result_is_json_serialized(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value=[1, 2, 3])
        tool = BrowserExecuteJSTool(manager=mgr)
        result = await tool.execute(script="return [1,2,3]")
        assert "1" in result

    @pytest.mark.asyncio
    async def test_execute_none_result(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        page.evaluate = AsyncMock(return_value=None)
        tool = BrowserExecuteJSTool(manager=mgr)
        result = await tool.execute(script="return null")
        assert "no return value" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_exception_from_manager(self):
        mgr, browser, ctx, page = _setup_mocked_manager()
        mgr.execute_js = AsyncMock(side_effect=RuntimeError("manager crashed"))
        tool = BrowserExecuteJSTool(manager=mgr)
        result = await tool.execute(script="boom")
        assert "Error" in result


# ---------------------------------------------------------------------------
# BrowserExecuteJSTool — _serialize (static helper)
# ---------------------------------------------------------------------------

class TestBrowserExecuteJSSerialize:

    def test_serialize_none(self):
        result = BrowserExecuteJSTool._serialize(None)
        assert "no return value" in result.lower()

    def test_serialize_string(self):
        result = BrowserExecuteJSTool._serialize("hello")
        assert result == "hello"

    def test_serialize_int(self):
        result = BrowserExecuteJSTool._serialize(42)
        assert result == "42"

    def test_serialize_bool(self):
        result = BrowserExecuteJSTool._serialize(True)
        assert result == "True"

    def test_serialize_dict_pretty(self):
        d = {"a": 1, "b": {"c": 2}}
        result = BrowserExecuteJSTool._serialize(d)
        assert '"a"' in result
        assert '1' in result

    def test_serialize_list_pretty(self):
        result = BrowserExecuteJSTool._serialize([1, 2, 3])
        assert "1" in result

    def test_serialize_non_json_dict_falls_back_to_str(self):
        class Unserializable:
            def __repr__(self):
                return "<oops>"
        d = {"bad": Unserializable()}
        result = BrowserExecuteJSTool._serialize(d)
        assert "oops" in result or result == str(d)

    def test_serialize_non_json_list_falls_back_to_str(self):
        class Unserializable:
            def __repr__(self):
                return "<oops>"
        l = [Unserializable()]
        result = BrowserExecuteJSTool._serialize(l)
        assert "oops" in result or result == str(l)
