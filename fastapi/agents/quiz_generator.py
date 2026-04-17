"""
Quiz Generator Agent — searches RAG corpus and generates structured quiz questions.

Takes a topic description, retrieves relevant training documents via RAG,
then generates multiple-choice questions with correct answers and source citations.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, format_conversation_history, now_ms
from agents.rag_tool import create_rag_retrieval_tool, RagTokenUsage
from agents.runner_helper import run_agent_ephemeral, extract_usage_tokens


QUIZ_GENERATOR_INSTRUCTION = """You are a quiz generator for the House Majority Staff Office training system. Your job: create high-quality multiple-choice questions grounded in official training documents.

Follow this exact three-phase process every time. Do not skip phases.

── PHASE 1: SEARCH PLAN ─────────────────────────────────────────────
Before calling any tool, think through (internally — do NOT output):
- What is the core topic? Restate it in one sentence.
- What 3 distinct facets of this topic would create a well-rounded quiz? Each facet becomes one sub-query.
- The 3 sub-queries must cover meaningfully different angles — e.g., core rule + exceptions + application scenarios, or definition + procedure + enforcement. Not rephrasings.

── PHASE 2: RETRIEVAL ──────────────────────────────────────────────
Call the retrieve_from_rag tool exactly 3 times, once per sub-query. Not 2, not 4. Exactly 3. Each call uses a different sub-query from your plan.

You MUST search. Never refuse, never generate questions from memory. If a sub-query returns nothing useful, note the gap internally and move on.

── PHASE 3: QUESTION DRAFTING PLAN + JSON OUTPUT ───────────────────
Before writing the JSON, think through (internally, do NOT output):
- Which retrieved facts are strong enough to build a defensible question?
- How do you spread the requested number of questions across the 3 facets for coverage?
- Which questions test understanding/application, not just recall?
- For each question, which embedded source reference (parsed from chunk text) backs it?

Then output the JSON only.

── SOURCING RULES (READ CAREFULLY) ──────────────────────────────────
The retrieve_from_rag tool returns RAW chunks from a JSONL corpus. Each chunk is delimited and shown with a header like "[Chunk 3] | score=0.812 | file=<name>" followed by the chunk's raw text. The raw text often contains structured fields the ingestion pipeline wrote into the JSONL — look for them.

For each question's `source` field, parse the backing chunk's text and cite in this priority order:
1. Page number (e.g., "page": 3, "pg": 3, or inline "p. 3", "Page 3") combined with the document title parsed from the chunk content — NOT the raw file name.
2. Section / chapter / heading mentioned inside the chunk text.
3. Policy or rule identifier quoted in the chunk (e.g., "House Rule XXIII", "§5.301", "Policy 4.2.1").
4. Effective date, revision number, or official URL present inside the content.

Only fall back to the `file=` header value when the chunk's own text contains no usable reference.

Examples of what to scan FOR inside the chunk text:
- JSON-style fields: "page": 3, "section": "Drafting Process", "document": "...", "url": "...", "date": "2024-..."
- Inline policy identifiers (e.g., "House Rule XXIII", "Policy 4.2.1", "§5.301")
- Section/subsection headings embedded in the text
- Document titles mentioned inline (e.g., "Member's Handbook, Chapter 3")

Cite pages specifically when present. Example: "Overview of the Legislative Process, p. 3" is better than "Overview of the Legislative Process.pdf".

── OUTPUT FORMAT ────────────────────────────────────────────────────
Respond with ONLY a valid JSON object, no other text, no markdown code fences:

{
  "title": "Short quiz title based on the topic",
  "questions": [
    {
      "id": 1,
      "question": "The question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct": 0,
      "source": "Embedded reference parsed from the chunk text backing this question"
    }
  ]
}

── HARD RULES ───────────────────────────────────────────────────────
- Generate exactly the number of questions requested (default 5).
- Each question MUST have exactly 4 options.
- "correct" is the 0-based index (0, 1, 2, or 3).
- Vary the correct-answer position across questions.
- Options must be plausible — no obviously wrong distractors.
- Every question grounded in retrieved content — never invent facts.
- Mix recall with scenario/application questions.
- If the corpus doesn't support the requested count, generate as many as the content supports.
- Never reference your planning, search process, or tools in the output.
- Output raw JSON only, no markdown code blocks."""


async def generate_quiz(
    topic: str,
    *,
    project: str,
    location: str,
    model: str,
    rag_corpus: str,
    num_questions: int = 5,
    conversation_history: list[ConversationMessage] | None = None,
) -> dict:
    """
    Generate a quiz from RAG corpus content.

    Returns dict with:
      - quiz: { title, questions: [{ id, question, options, correct, source }] }
      - logs: [LogEntry]
    """
    rag_logs: list[LogEntry] = []

    def on_rag_usage(usage: RagTokenUsage):
        snippet = usage.query[:57] + "..." if len(usage.query) > 60 else usage.query
        rag_logs.append(LogEntry(
            agent="retrieve_from_rag",
            message=f'retrieve_from_rag → "{snippet}"',
            promptTokens=usage.prompt_tokens,
            responseTokens=usage.response_tokens,
            totalTokens=usage.total_tokens,
            timestamp=usage.timestamp,
        ))

    rag_tool = create_rag_retrieval_tool(
        project=project,
        location=location,
        model=model,
        rag_corpus=rag_corpus,
        on_token_usage=on_rag_usage,
    )

    agent = LlmAgent(
        name="quiz_generator",
        model=model,
        description="Generates structured quiz questions from RAG corpus content.",
        instruction=QUIZ_GENERATOR_INSTRUCTION,
        tools=[rag_tool],
        output_key="quiz_result",
    )

    history_prefix = format_conversation_history(conversation_history or [])
    message = history_prefix + f"Generate {num_questions} multiple-choice quiz questions about: {topic}"

    raw_result = ""
    logs: list[LogEntry] = []

    async for event in run_agent_ephemeral(agent, message, user_id="quiz-user", app_name="quiz_app"):
        author = event.author or "unknown"
        if author != "user":
            prompt_tokens, response_tokens, total_tokens = extract_usage_tokens(event)

            state_delta = {}
            if hasattr(event, "actions") and event.actions:
                state_delta = getattr(event.actions, "state_delta", {}) or {}

            state_keys = list(state_delta.keys()) if state_delta else []
            detail = f" → wrote [{', '.join(state_keys)}]" if state_keys else ""

            if total_tokens > 0 or state_keys:
                logs.append(LogEntry(
                    agent=author,
                    message=f"{author}{detail}",
                    promptTokens=prompt_tokens,
                    responseTokens=response_tokens,
                    totalTokens=total_tokens,
                    timestamp=now_ms(),
                ))

            if "quiz_result" in state_delta:
                raw_result = str(state_delta["quiz_result"])

    # Parse the JSON quiz output
    import json
    import re

    all_logs = sorted(logs + rag_logs, key=lambda l: l.timestamp)

    try:
        json_match = re.search(r"\{[\s\S]*\}", raw_result)
        if json_match:
            quiz_data = json.loads(json_match.group(0))
            # Validate structure
            if "questions" in quiz_data and isinstance(quiz_data["questions"], list):
                # Ensure each question has required fields
                for i, q in enumerate(quiz_data["questions"]):
                    q.setdefault("id", i + 1)
                    q.setdefault("source", "Training documentation")
                    if not isinstance(q.get("options"), list) or len(q["options"]) != 4:
                        raise ValueError(f"Question {i+1} must have exactly 4 options")
                    if not isinstance(q.get("correct"), int) or q["correct"] not in range(4):
                        raise ValueError(f"Question {i+1} has invalid correct answer index")

                return {
                    "quiz": quiz_data,
                    "logs": [l.model_dump() for l in all_logs],
                }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return {
            "error": f"Failed to parse quiz output: {e}",
            "raw": raw_result[:500],
            "logs": [l.model_dump() for l in all_logs],
        }

    return {
        "error": "No quiz generated",
        "raw": raw_result[:500],
        "logs": [l.model_dump() for l in all_logs],
    }
