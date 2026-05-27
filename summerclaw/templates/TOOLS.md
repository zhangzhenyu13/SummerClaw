# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## glob — File Discovery

- Use `glob` to find files by pattern before falling back to shell commands
- Simple patterns like `*.py` match recursively by filename
- Use `entry_type="dirs"` when you need matching directories instead of files
- Use `head_limit` and `offset` to page through large result sets
- Prefer this over `exec` when you only need file paths

## grep — Content Search

- Use `grep` to search file contents — **scope searches to `memory/` or `outputs/` only**, never the workspace root
- Default behavior returns only matching file paths (`output_mode="files_with_matches"`)
- Supports optional `glob` filtering plus `context_before` / `context_after`
- Supports `type="py"`, `type="ts"`, `type="md"` and similar shorthand filters
- Use `fixed_strings=true` for literal keywords containing regex characters
- Use `output_mode="files_with_matches"` to get only matching file paths
- Use `output_mode="count"` to size a search before reading full matches
- Use `head_limit` and `offset` to page across results
- Prefer this over `exec` for code and history searches
- Binary or oversized files may be skipped to keep results readable

## write_file — File Organization (CRITICAL)

- **NEVER create files directly in workspace root** (except AGENTS.md, SOUL.md, USER.md, TOOLS.md, HEARTBEAT.md)
- **ALWAYS use `outputs/<project-name>/` for user work products**
  - Example: `outputs/f1-grand-prix/index.html`
  - Example: `outputs/raiden4/scripts/game.js`
  - Example: `outputs/reports/analysis.md`
- The `write_file` tool enforces this — non-system files in workspace root will be **rejected**
- Each write to `outputs/` is automatically recorded in `outputs/meta.json` with timestamp and source context
- This keeps the workspace organized and prevents clutter

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
