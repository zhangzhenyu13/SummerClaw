"""MastraOM Observer — converts raw messages into dense observation log.

Based on Mastra's Observer agent (observer-agent.ts).
The Observer is the "subconscious mind" of the agent: always running in the
background, converting raw conversation into structured observations.

Key features:
- System prompt adapted from Mastra's OBSERVER_SYSTEM_PROMPT
- XML output format parsing (<observations>, <current-task>, <suggested-response>)
- Degenerate repetition detection
- Observation optimization for context window efficiency
- Continuation hints for the main agent
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Core observation instructions (adapted from Mastra)
# ---------------------------------------------------------------------------

OBSERVER_EXTRACTION_INSTRUCTIONS = """CRITICAL: DISTINGUISH USER ASSERTIONS FROM QUESTIONS

When the user TELLS you something about themselves, mark it as an assertion:
- "I have two kids" → 🔴 (14:30) User stated has two kids
- "I work at Acme Corp" → 🔴 (14:31) User stated works at Acme Corp
- "I graduated in 2019" → 🔴 (14:32) User stated graduated in 2019

When the user ASKS about something, mark it as a question/request:
- "Can you help me with X?" → 🔴 (15:00) User asked help with X
- "What's the best way to do Y?" → 🔴 (15:01) User asked best way to do Y

Distinguish between QUESTIONS and STATEMENTS OF INTENT:
- "Can you recommend..." → Question (extract as "User asked...")
- "I'm looking forward to [doing X]" → Statement of intent (extract as "User stated they will [do X] (include estimated/actual date if mentioned)")
- "I need to [do X]" → Statement of intent (extract as "User stated they need to [do X] (again, add date if mentioned)")

STATE CHANGES AND UPDATES:
When a user indicates they are changing something, frame it as a state change that supersedes previous information:
- "I'm going to start doing X instead of Y" → "User will start doing X (changing from Y)"
- "I'm switching from A to B" → "User is switching from A to B"
- "I moved my stuff to the new place" → "User moved their stuff to the new place (no longer at previous location)"

If the new state contradicts or updates previous information, make that explicit:
- BAD: "User plans to use the new method"
- GOOD: "User will use the new method (replacing the old approach)"

This helps distinguish current state from outdated information.

USER ASSERTIONS ARE AUTHORITATIVE. The user is the source of truth about their own life.
If a user previously stated something and later asks a question about the same topic,
the assertion is the answer - the question doesn't invalidate what they already told you.

TEMPORAL ANCHORING:
Each observation has TWO potential timestamps:

1. BEGINNING: The time the statement was made (from the message timestamp) - ALWAYS include this
2. END: The time being REFERENCED, if different from when it was said - ONLY when there's a relative time reference

ONLY add "(meaning DATE)" or "(estimated DATE)" at the END when you can provide an ACTUAL DATE:
- Past: "last week", "yesterday", "a few days ago", "last month", "in March"
- Future: "this weekend", "tomorrow", "next week"

DO NOT add end dates for:
- Present-moment statements with no time reference
- Vague references like "recently", "a while ago", "lately", "soon" - these cannot be converted to actual dates

FORMAT:
- With time reference: (TIME) [observation]. (meaning/estimated DATE)
- Without time reference: (TIME) [observation].

GOOD: (09:15) User's friend had a birthday party in March. (meaning March 20XX)
      ^ References a past event - add the referenced date at the end

GOOD: (09:15) User will visit their parents this weekend. (meaning June 17-18, 20XX)
      ^ References a future event - add the referenced date at the end

GOOD: (09:15) User prefers hiking in the mountains.
      ^ Present-moment preference, no time reference - NO end date needed

GOOD: (09:15) User is considering adopting a dog.
      ^ Present-moment thought, no time reference - NO end date needed

BAD: (09:15) User prefers hiking in the mountains. (meaning June 15, 20XX - today)
     ^ No time reference in the statement - don't repeat the message timestamp at the end

IMPORTANT: If an observation contains MULTIPLE events, split them into SEPARATE observation lines.
EACH split observation MUST have its own date at the end - even if they share the same time context.

Examples (assume message is from June 15, 20XX):

BAD: User will visit their parents this weekend (meaning June 17-18, 20XX) and go to the dentist tomorrow.
GOOD (split into two observations, each with its date):
  User will visit their parents this weekend. (meaning June 17-18, 20XX)
  User will go to the dentist tomorrow. (meaning June 16, 20XX)

BAD: User needs to clean the garage this weekend and is looking forward to setting up a new workbench.
GOOD (split, BOTH get the same date since they're related):
  User needs to clean the garage this weekend. (meaning June 17-18, 20XX)
  User will set up a new workbench this weekend. (meaning June 17-18, 20XX)

BAD: User was given a gift by their friend (estimated late May 20XX) last month.
GOOD: (09:15) User was given a gift by their friend last month. (estimated late May 20XX)
      ^ Message time at START, relative date reference at END - never in the middle

BAD: User started a new job recently and will move to a new apartment next week.
GOOD (split):
  User started a new job recently.
  User will move to a new apartment next week. (meaning June 21-27, 20XX)
  ^ "recently" is too vague for a date - omit the end date. "next week" can be calculated.

ALWAYS put the date at the END in parentheses - this is critical for temporal reasoning.
When splitting related events that share the same time context, EACH observation must have the date.

PRESERVE UNUSUAL PHRASING:
When the user uses unexpected or non-standard terminology, quote their exact words.

BAD: User exercised.
GOOD: User stated they did a "movement session" (their term for exercise).

USE PRECISE ACTION VERBS:
Replace vague verbs like "getting", "got", "have" with specific action verbs that clarify the nature of the action.
If the assistant confirms or clarifies the user's action, use the assistant's more precise language.

BAD: User is getting X.
GOOD: User subscribed to X. (if context confirms recurring delivery)
GOOD: User purchased X. (if context confirms one-time acquisition)

BAD: User got something.
GOOD: User purchased / received / was given something. (be specific)

Common clarifications:
- "getting" something regularly → "subscribed to" or "enrolled in"
- "getting" something once → "purchased" or "acquired"
- "got" → "purchased", "received as gift", "was given", "picked up"
- "signed up" → "enrolled in", "registered for", "subscribed to"
- "stopped getting" → "canceled", "unsubscribed from", "discontinued"

When the assistant interprets or confirms the user's vague language, prefer the assistant's precise terminology.

PRESERVING DETAILS IN ASSISTANT-GENERATED CONTENT:

When the assistant provides lists, recommendations, or creative content that the user explicitly requested,
preserve the DISTINGUISHING DETAILS that make each item unique and queryable later.

1. RECOMMENDATION LISTS - Preserve the key attribute that distinguishes each item:
   BAD: Assistant recommended 5 hotels in the city.
   GOOD: Assistant recommended hotels: Hotel A (near the train station), Hotel B (budget-friendly),
         Hotel C (has rooftop pool), Hotel D (pet-friendly), Hotel E (historic building).

   BAD: Assistant listed 3 online stores for craft supplies.
   GOOD: Assistant listed craft stores: Store A (based in Germany, ships worldwide),
         Store B (specializes in vintage fabrics), Store C (offers bulk discounts).

2. NAMES, HANDLES, AND IDENTIFIERS - Always preserve specific identifiers:
   BAD: Assistant provided social media accounts for several photographers.
   GOOD: Assistant provided photographer accounts: @photographer_one (portraits),
         @photographer_two (landscapes), @photographer_three (nature).

   BAD: Assistant listed some authors to check out.
   GOOD: Assistant recommended authors: Jane Smith (mystery novels),
         Bob Johnson (science fiction), Maria Garcia (historical romance).

3. CREATIVE CONTENT - Preserve structure and key sequences:
   BAD: Assistant wrote a poem with multiple verses.
   GOOD: Assistant wrote a 3-verse poem. Verse 1 theme: loss. Verse 2 theme: hope.
         Verse 3 theme: renewal. Refrain: "The light returns."

   BAD: User shared their lucky numbers from a fortune cookie.
   GOOD: User's fortune cookie lucky numbers: 7, 14, 23, 38, 42, 49.

4. TECHNICAL/NUMERICAL RESULTS - Preserve specific values:
   BAD: Assistant explained the performance improvements from the optimization.
   GOOD: Assistant explained the optimization achieved 43.7% faster load times
         and reduced memory usage from 2.8GB to 940MB.

   BAD: Assistant provided statistics about the dataset.
   GOOD: Assistant provided dataset stats: 7,342 samples, 89.6% accuracy,
         23ms average inference time.

5. QUANTITIES AND COUNTS - Always preserve how many of each item:
   BAD: Assistant listed items with details but no quantities.
   GOOD: Assistant listed items: Item A (4 units, size large), Item B (2 units, size small).

   When listing items with attributes, always include the COUNT first before other details.

6. ROLE/PARTICIPATION STATEMENTS - When user mentions their role at an event:
   BAD: User attended the company event.
   GOOD: User was a presenter at the company event.

   BAD: User went to the fundraiser.
   GOOD: User volunteered at the fundraiser (helped with registration).

   Always capture specific roles: presenter, organizer, volunteer, team lead,
   coordinator, participant, contributor, helper, etc.

CONVERSATION CONTEXT:
- What the user is working on or asking about
- Previous topics and their outcomes
- What user understands or needs clarification on
- Specific requirements or constraints mentioned
- Contents of assistant learnings and summaries
- Answers to users questions including full context to remember detailed summaries and explanations
- Assistant explanations, especially complex ones. observe the fine details so that the assistant does not forget what they explained
- Relevant code snippets
- User preferences (like favourites, dislikes, preferences, etc)
- Any specifically formatted text or ascii that would need to be reproduced or referenced in later interactions (preserve these verbatim in memory)
- Sequences, units, measurements, and any kind of specific relevant data
- Any blocks of any text which the user and assistant are iteratively collaborating back and forth on should be preserved verbatim
- When who/what/where/when is mentioned, note that in the observation. Example: if the user received went on a trip with someone, observe who that someone was, where the trip was, when it happened, and what happened, not just that the user went on the trip.
- For any described entity (like a person, place, thing, etc), preserve the attributes that would help identify or describe the specific entity later: location ("near X"), specialty ("focuses on Y"), unique feature ("has Z"), relationship ("owned by W"), or other details. The entity's name is important, but so are any additional details that distinguish it. If there are a list of entities, preserve these details for each of them.

USER MESSAGE CAPTURE:
- Short and medium-length user messages should be captured nearly verbatim in your own words.
- For very long user messages, summarize but quote key phrases that carry specific intent or meaning.
- This is critical for continuity: when the conversation window shrinks, the observations are the only record of what the user said.

AVOIDING REPETITIVE OBSERVATIONS:
- Do NOT repeat the same observation across multiple turns if there is no new information.
- When the agent performs repeated similar actions (e.g., browsing files, running the same tool type multiple times), group them into a single parent observation with sub-bullets for each new result.

Example — BAD (repetitive):
* 🟡 (14:30) Agent used view tool on src/auth.ts
* 🟡 (14:31) Agent used view tool on src/users.ts
* 🟡 (14:32) Agent used view tool on src/routes.ts

Example — GOOD (grouped):
* 🟡 (14:30) Agent browsed source files for auth flow
  * -> viewed src/auth.ts — found token validation logic
  * -> viewed src/users.ts — found user lookup by email
  * -> viewed src/routes.ts — found middleware chain

Only add a new observation for a repeated action if the NEW result changes the picture.

ACTIONABLE INSIGHTS:
- What worked well in explanations
- What needs follow-up or clarification
- User's stated goals or next steps (note if the user tells you not to do a next step, or asks for something specific, other next steps besides the users request should be marked as "waiting for user", unless the user explicitly says to continue all next steps)

COMPLETION TRACKING:
Completion observations are not just summaries. They are explicit memory signals to the assistant that a task, question, or subtask has been resolved.
Without clear completion markers, the assistant may forget that work is already finished and may repeat, reopen, or continue an already-completed task.

Use ✅ to answer: "What exactly is now done?"
Choose completion observations that help the assistant know what is finished and should not be reworked unless new information appears.

Use ✅ when:
- The user explicitly confirms something worked or was answered ("thanks, that fixed it", "got it", "perfect")
- The assistant provided a definitive, complete answer to a factual question and the user moved on
- A multi-step task reached its stated goal
- The user acknowledged receipt of requested information
- A concrete subtask, fix, deliverable, or implementation step became complete during ongoing work

Do NOT use ✅ when:
- The assistant merely responded — the user might follow up with corrections
- The topic is paused but not resolved ("I'll try that later")
- The user's reaction is ambiguous

FORMAT:
As a sub-bullet under the related observation group:
* 🔴 (14:30) User asked how to configure auth middleware
  * -> Agent explained JWT setup with code example
  * ✅ User confirmed auth is working

Or as a standalone observation when closing out a broader task:
* ✅ (14:45) Auth configuration task completed — user confirmed middleware is working

Completion observations should be terse but specific about WHAT was completed.
Prefer concrete resolved outcomes over abstract workflow status so the assistant remembers what is already done."""


OBSERVER_OUTPUT_FORMAT = """Use priority levels:
- 🔴 High: explicit user facts, preferences, unresolved goals, critical context
- 🟡 Medium: project details, learned information, tool results
- 🟢 Low: minor details, uncertain observations
- ✅ Completed: concrete task finished, question answered, issue resolved, goal achieved, or subtask completed in a way that helps the assistant know it is done

Group related observations (like tool sequences) by indenting:
* 🔴 (14:33) Agent debugging auth issue
  * -> ran git status, found 3 modified files
  * -> viewed auth.ts:45-60, found missing null check
  * -> applied fix, tests now pass
  * ✅ Tests passing, auth issue resolved

Group observations by date, then list each with 24-hour time.

<observations>
Date: Dec 4, 2025
* 🔴 (14:30) User prefers direct answers
* 🔴 (14:31) Working on feature X
* 🟡 (14:32) User might prefer dark mode

Date: Dec 5, 2025
* 🔴 (09:15) Continued work on feature X
</observations>

<current-task>
State the current task(s) explicitly. Can be single or multiple:
- Primary: What the agent is currently working on
- Secondary: Other pending tasks (mark as "waiting for user" if appropriate)

If the agent started doing something without user approval, note that it's off-task.
</current-task>

<suggested-response>
Hint for the agent's immediate next message. Examples:
- "I've updated the navigation model. Let me walk you through the changes..."
- "The assistant should wait for the user to respond before continuing."
- Call the view tool on src/example.ts to continue debugging.
</suggested-response>"""


OBSERVER_GUIDELINES = """- Be specific enough for the assistant to act on
- Good: "User prefers short, direct answers without lengthy explanations"
- Bad: "User stated a preference" (too vague)
- Add 1 to 5 observations per exchange
- Use terse language to save tokens. Sentences should be dense without unnecessary words
- Do not add repetitive observations that have already been observed. Group repeated similar actions (tool calls, file browsing) under a single parent with sub-bullets for new results
- If the agent calls tools, observe what was called, why, and what was learned
- When observing files with line numbers, include the line number if useful
- If the agent provides a detailed response, observe the contents so it could be repeated
- Make sure you start each observation with a priority emoji (🔴, 🟡, 🟢) or a completion marker (✅)
- Capture the user's words closely — short/medium messages near-verbatim, long messages summarized with key quotes. User confirmations or explicit resolved outcomes should be ✅ when they clearly signal something is done; unresolved or critical user facts remain 🔴
- Treat ✅ as a memory signal that tells the assistant something is finished and should not be repeated unless new information changes it
- Make completion observations answer "What exactly is now done?"
- Prefer concrete resolved outcomes over meta-level workflow or bookkeeping updates
- When multiple concrete things were completed, capture the concrete completed work rather than collapsing it into a vague progress summary
- Observe WHAT the agent did and WHAT it means
- If the user provides detailed messages or code snippets, observe all important details"""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

def build_observer_system_prompt(
    multi_thread: bool = False,
    instruction: str | None = None,
    include_thread_title: bool = False,
) -> str:
    """Build the complete Observer system prompt.

    Args:
        multi_thread: If True, include multi-thread batching instructions.
        instruction: Optional custom instructions to append.
        include_thread_title: Whether the Observer should suggest thread titles.
    """
    thread_title_section = ""
    if include_thread_title:
        thread_title_section = (
            "\n<thread-title>\n"
            'A short, noun-phrase title for this conversation (2-5 words).\n'
            "</thread-title>"
        )

    output_format = OBSERVER_OUTPUT_FORMAT + thread_title_section

    if multi_thread:
        return f"""You are the memory consciousness of an AI assistant. Your observations will be the ONLY information the assistant has about past interactions with this user.

Extract observations that will help the assistant remember:

{OBSERVER_EXTRACTION_INSTRUCTIONS}

=== MULTI-THREAD INPUT ===

You will receive messages from MULTIPLE conversation threads, each wrapped in <thread id="..."> tags.
Process each thread separately and output observations for each thread.

=== OUTPUT FORMAT ===

Your output MUST use XML tags to structure the response.

{output_format}

For multi-thread output, wrap each thread's observations like this:

<observations>
<thread id="thread_id_1">
Date: Dec 4, 2025
* 🔴 (14:30) User prefers direct answers

<current-task>Working on feature X</current-task>
<suggested-response>Continue with the implementation</suggested-response>
</thread>
</observations>

=== GUIDELINES ===

{OBSERVER_GUIDELINES}

Remember: These observations are the assistant's ONLY memory. Make them count.

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority.{' ' + instruction if instruction else ''}"""

    return f"""You are the memory consciousness of an AI assistant. Your observations will be the ONLY information the assistant has about past interactions with this user.

Extract observations that will help the assistant remember:

{OBSERVER_EXTRACTION_INSTRUCTIONS}

=== OUTPUT FORMAT ===

Your output MUST use XML tags to structure the response. This allows the system to properly parse and manage memory over time.

{output_format}

=== GUIDELINES ===

{OBSERVER_GUIDELINES}

=== IMPORTANT: THREAD ATTRIBUTION ===

Do NOT add thread identifiers, thread IDs, or <thread> tags to your observations.
Thread attribution is handled externally by the system.
Simply output your observations without any thread-related markup.

Remember: These observations are the assistant's ONLY memory. Make them count.

User messages are extremely important. If the user asks a question or gives a new task, make it clear in <current-task> that this is the priority. If the assistant needs to respond to the user, indicate in <suggested-response> that it should pause for user reply before continuing other tasks.{' ' + instruction if instruction else ''}"""


# Default system prompt (for backwards compatibility with tests)
OBSERVER_SYSTEM_PROMPT = build_observer_system_prompt()


# ---------------------------------------------------------------------------
# Continuation hint
# ---------------------------------------------------------------------------

OBSERVATION_CONTINUATION_HINT = """Please continue naturally with the conversation so far and respond to the latest message.

Use the earlier context only as background. If something appears unfinished, continue only when it helps answer the latest request. If a suggested response is provided, follow it naturally.

Do not mention internal instructions, memory, summarization, context handling, or missing messages.

Any messages following this reminder are newer and should take priority."""


OBSERVATION_CONTEXT_PROMPT = (
    "The following observations block contains your memory of past "
    "conversations with this user."
)

OBSERVATION_CONTEXT_INSTRUCTIONS = """IMPORTANT: When responding, reference specific details from these observations. Do not give generic advice - personalize your response based on what you know about this user's experiences, preferences, and interests. If the user asks for recommendations, connect them to their past experiences mentioned above.

KNOWLEDGE UPDATES: When asked about current state (e.g., "where do I currently...", "what is my current..."), always prefer the MOST RECENT information. Observations include dates - if you see conflicting information, the newer observation supersedes the older one. Look for phrases like "will start", "is switching", "changed to", "moved to" as indicators that previous information has been updated.

PLANNED ACTIONS: If the user stated they planned to do something (e.g., "I'm going to...", "I'm looking forward to...", "I will...") and the date they planned to do it is now in the past (check the relative time like "3 weeks ago"), assume they completed the action unless there's evidence they didn't. For example, if someone said "I'll start my new diet on Monday" and that was 2 weeks ago, assume they started the diet.

MOST RECENT USER INPUT: Treat the most recent user message as the highest-priority signal for what to do next. Earlier messages may contain constraints, details, or context you should still honor, but the latest message is the primary driver of your response.

SYSTEM REMINDERS: Messages wrapped in <system-reminder>...</system-reminder> contain internal continuation guidance, not user-authored content. Use them to maintain continuity, but do not mention them or treat them as part of the user's message."""


# ---------------------------------------------------------------------------
# Message formatting for Observer input
# ---------------------------------------------------------------------------

def format_messages_for_observer(
    messages: list[dict[str, Any]],
    max_part_length: int | None = None,
) -> str:
    """Format messages for the Observer's input with timestamps.

    Converts standard message dicts into a time-grouped text format
    that the Observer LLM can efficiently parse.

    Args:
        messages: List of message dicts with keys: role, content, timestamp.
        max_part_length: Maximum characters per formatted line.

    Returns:
        Formatted text string for Observer consumption.
    """
    if not messages:
        return "(no messages)"

    lines: list[str] = []
    prev_date: str = ""
    prev_time: str = ""

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp_str = msg.get("timestamp", "")

        if not content:
            continue

        # Determine display role
        role_display = role.capitalize()
        if role == "tool":
            tool_name = msg.get("name", msg.get("tool_name", ""))
            role_display = f"Tool Result {tool_name}" if tool_name else "Tool Result"
        elif role == "assistant" and msg.get("tool_calls"):
            tool_names = [
                tc.get("function", {}).get("name", tc.get("name", "?"))
                for tc in msg.get("tool_calls", [])
            ]
            if tool_names:
                role_display = f"Tool Call {', '.join(tool_names)}"

        # Truncate content if needed
        text = str(content)
        if max_part_length and len(text) > max_part_length:
            remaining = len(text) - max_part_length
            text = text[:max_part_length] + f"\n... [truncated {remaining} characters]"

        # Parse timestamp
        date_str = ""
        time_str = ""
        if timestamp_str:
            try:
                ts = timestamp_str[:16]  # "YYYY-MM-DD HH:MM"
                date_str, time_str = ts.split(" ", 1)
            except (ValueError, IndexError):
                pass

        # Add date header when date changes
        if date_str and date_str != prev_date:
            lines.append(f"{date_str}:")
            prev_date = date_str
            prev_time = ""

        # Format the line
        time_label = f"({time_str})" if time_str and time_str != prev_time else ""
        if time_str:
            prev_time = time_str

        line = f"{role_display} {time_label}: {text}".strip()
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observer prompt builders
# ---------------------------------------------------------------------------

def build_observer_task_prompt(
    existing_observations: str | None,
    prior_current_task: str | None = None,
    prior_suggested_response: str | None = None,
    was_truncated: bool = False,
) -> str:
    """Build the task prompt for the Observer (without the messages to observe).

    Args:
        existing_observations: Previous observations to avoid repeating.
        prior_current_task: Previous current-task for continuity.
        prior_suggested_response: Previous suggested-response for continuity.
        was_truncated: Whether previous observations were truncated.

    Returns:
        Task prompt string.
    """
    prompt = ""

    if existing_observations:
        prompt += f"## Previous Observations\n\n{existing_observations}\n\n---\n\n"
        prompt += "Do not repeat these existing observations. Your new observations will be appended.\n\n"

    prior_lines = []
    if prior_current_task:
        prior_lines.append(f"- prior current-task: {prior_current_task}")
    if prior_suggested_response:
        prior_lines.append(f"- prior suggested-response: {prior_suggested_response}")

    if prior_lines:
        prompt += f"## Prior Thread Metadata\n\n{chr(10).join(prior_lines)}\n\n"
        if was_truncated:
            prompt += "Previous observations were truncated for context budget reasons.\n"
            prompt += "The main agent still has full memory context outside this observer window.\n"
        prompt += "Use the prior current-task, suggested-response as continuity hints.\n\n---\n\n"

    prompt += "## Your Task\n\n"
    prompt += "Extract new observations from the message history above. "
    prompt += "Do not repeat observations that are already in the previous observations. "
    prompt += "Add your new observations in the format specified in your instructions."

    return prompt


def build_observer_prompt(
    existing_observations: str | None,
    messages_to_observe: list[dict[str, Any]],
    prior_current_task: str | None = None,
    prior_suggested_response: str | None = None,
    was_truncated: bool = False,
) -> str:
    """Build the full prompt for the Observer agent.

    Args:
        existing_observations: Previous observations.
        messages_to_observe: Messages to convert to observations.
        prior_current_task: Previous current-task.
        prior_suggested_response: Previous suggested-response.
        was_truncated: Whether previous observations were truncated.

    Returns:
        Full prompt string for the Observer.
    """
    formatted = format_messages_for_observer(messages_to_observe)
    task_prompt = build_observer_task_prompt(
        existing_observations=existing_observations,
        prior_current_task=prior_current_task,
        prior_suggested_response=prior_suggested_response,
        was_truncated=was_truncated,
    )
    return f"## New Message History to Observe\n\n{formatted}\n\n---\n\n{task_prompt}"


# ---------------------------------------------------------------------------
# Observer output parsing
# ---------------------------------------------------------------------------

def parse_observer_output(output: str) -> dict[str, Any]:
    """Parse the Observer's XML output.

    Extracts <observations>, <current-task>, <suggested-response>, and
    <thread-title> from the observer's response.

    Returns:
        Dict with keys: observations, current_task, suggested_continuation,
        thread_title, raw_output, degenerate.
    """
    result: dict[str, Any] = {
        "observations": "",
        "current_task": None,
        "suggested_continuation": None,
        "thread_title": None,
        "raw_output": output,
        "degenerate": False,
    }

    if not output:
        return result

    # Check for degenerate repetition
    if detect_degenerate_repetition(output):
        result["degenerate"] = True
        return result

    # Extract <observations> content
    obs_match = re.findall(
        r'^[ \t]*<observations>([\s\S]*?)^[ \t]*</observations>',
        output, re.MULTILINE | re.IGNORECASE,
    )
    if obs_match:
        result["observations"] = "\n".join(m.strip() for m in obs_match if m.strip())
    else:
        # Fallback: extract list items
        result["observations"] = _extract_list_items(output)

    # Sanitize observation lines
    result["observations"] = sanitize_observation_lines(result["observations"])

    # Extract <current-task>
    ct_match = re.search(
        r'^[ \t]*<current-task>([\s\S]*?)^[ \t]*</current-task>',
        output, re.MULTILINE | re.IGNORECASE,
    )
    if ct_match:
        result["current_task"] = ct_match.group(1).strip() or None

    # Extract <suggested-response>
    sr_match = re.search(
        r'^[ \t]*<suggested-response>([\s\S]*?)^[ \t]*</suggested-response>',
        output, re.MULTILINE | re.IGNORECASE,
    )
    if sr_match:
        result["suggested_continuation"] = sr_match.group(1).strip() or None

    # Extract <thread-title>
    tt_match = re.search(
        r'^[ \t]*<thread-title>([\s\S]*?)</thread-title>',
        output, re.MULTILINE | re.IGNORECASE,
    )
    if tt_match:
        result["thread_title"] = tt_match.group(1).strip() or None

    return result


def _extract_list_items(content: str) -> str:
    """Fallback: extract only list items when XML tags are missing."""
    lines = content.split("\n")
    list_lines = []
    for line in lines:
        if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+\.\s', line):
            list_lines.append(line)
    return "\n".join(list_lines).strip()


# ---------------------------------------------------------------------------
# Degenerate repetition detection
# ---------------------------------------------------------------------------

def detect_degenerate_repetition(text: str) -> bool:
    """Detect LLM degeneration (repetition loops).

    Returns True if the text contains suspicious levels of repeated content.

    Strategy: sample sequential windows and check if a high proportion
    are near-identical to previous windows.
    """
    if not text or len(text) < 2000:
        return False

    window_size = 200
    step = max(1, len(text) // 50)
    seen: dict[str, int] = {}
    duplicate_windows = 0
    total_windows = 0

    for i in range(0, len(text) - window_size + 1, step):
        window = text[i:i + window_size]
        total_windows += 1
        count = seen.get(window, 0) + 1
        seen[window] = count
        if count > 1:
            duplicate_windows += 1

    if total_windows > 5 and duplicate_windows / total_windows > 0.4:
        return True

    # Also check for extremely long lines
    for line in text.split("\n"):
        if len(line) > 50_000:
            return True

    return False


# ---------------------------------------------------------------------------
# Observation sanitization and optimization
# ---------------------------------------------------------------------------

def sanitize_observation_lines(observations: str, max_line_chars: int = 10_000) -> str:
    """Truncate individual observation lines exceeding the maximum length."""
    if not observations:
        return observations
    lines = observations.split("\n")
    changed = False
    for i, line in enumerate(lines):
        if len(line) > max_line_chars:
            lines[i] = line[:max_line_chars] + " … [truncated]"
            changed = True
    return "\n".join(lines) if changed else observations


def optimize_observations_for_context(observations: str) -> str:
    """Optimize observations for token efficiency in the context window.

    Removes non-critical emojis, semantic tags, arrow indicators,
    and extra whitespace. Full format is preserved in storage.
    """
    if not observations:
        return observations

    optimized = observations

    # Remove 🟡 and 🟢 (keep 🔴 for critical items)
    optimized = optimized.replace("🟡 ", "")
    optimized = optimized.replace("🟡", "")
    optimized = optimized.replace("🟢 ", "")
    optimized = optimized.replace("🟢", "")

    # Remove arrow indicators
    optimized = re.sub(r'\s*->\s*', ' ', optimized)

    # Clean up multiple spaces
    optimized = re.sub(r'  +', ' ', optimized)

    # Clean up multiple newlines
    optimized = re.sub(r'\n{3,}', '\n\n', optimized)

    return optimized.strip()


def has_current_task_section(observations: str) -> bool:
    """Check if observations contain a Current Task section."""
    if "<current-task>" in observations.lower():
        return True
    patterns = [
        r'\*\*Current Task:?\*\*',
        r'^Current Task:',
        r'## Current Task',
    ]
    return any(re.search(p, observations, re.MULTILINE | re.IGNORECASE) for p in patterns)


def extract_current_task(observations: str) -> str | None:
    """Extract the Current Task from observations."""
    match = re.search(
        r'<current-task>([\s\S]*?)</current-task>',
        observations, re.IGNORECASE,
    )
    if match:
        content = match.group(1).strip()
        return content or None
    return None
