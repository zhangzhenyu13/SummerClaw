# Anti-Infinite-Loop Guard

## Overview
Prevents AI agents from getting stuck in repetitive tool call loops, ensuring tasks fail gracefully or pivot to alternative strategies when progress stalls.

## Detection Heuristics
Trigger the fallback protocol if ANY of the following conditions are met:
- **Repetition Threshold**: The same tool is called ≥3 times with identical or near-identical parameters without a change in output state.
- **Progress Stagnation**: After 5 consecutive tool calls, the core objective has not advanced (e.g., no new data collected, no file modified, no error resolved).
- **Tool Failure Loop**: A specific tool fails ≥2 times consecutively with the same error message.

## Prevention Rules
1. **Explicit Max Retries**: Never retry a failing tool more than 2 times without changing parameters or strategy.
2. **State Validation**: After each tool call, explicitly verify if the output moves the task closer to completion. If not, log the stall and reassess.
3. **Alternative Routing**: If Tool A fails twice, immediately switch to Tool B (if available) or request user input. Do not retry Tool A blindly.
4. **Hard Stop Condition**: Define a clear "success" and "failure" state before starting. If neither is reached within 15 tool calls, halt and report.

## Fallback Protocol
When a loop is detected:
1. **STOP** calling the problematic tool immediately.
2. **DIAGNOSE**: Summarize the exact error, repetition count, and last known state.
3. **REPORT**: Inform the user: `"Task stalled after X attempts due to [Error]. Switching to fallback strategy."`
4. **PIVOT**: 
   - Try a different tool/method.
   - Ask the user for clarification or manual intervention.
   - Return partial results if available.

## Usage
- Apply this skill to any task involving web scraping, file processing, or multi-step automation.
- Integrate into system prompts or agent instructions as a mandatory safety layer.
- Review tool call logs periodically to tune repetition thresholds.
