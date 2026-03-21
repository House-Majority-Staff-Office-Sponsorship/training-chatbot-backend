"""
Quiz Generator Agent — searches RAG corpus and generates structured quiz questions.

Takes a topic description, retrieves relevant training documents via RAG,
then generates multiple-choice questions with correct answers and source citations.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from models import ConversationMessage, LogEntry, format_conversation_history, now_ms
from agents.rag_tool import create_rag_retrieval_tool, RagTokenUsage
from agents.runner_helper import run_agent_ephemeral


QUIZ_GENERATOR_INSTRUCTION = """You are a quiz generator for the House Majority Staff Office training system. Your job is to create high-quality multiple-choice quiz questions based on official training documents.

Your process:
1. Analyze the user's topic request and formulate 2-3 targeted search queries.
2. Call the retrieve_from_rag tool for EACH query to gather relevant information.
3. Based on the retrieved information, generate quiz questions.

CRITICAL OUTPUT FORMAT — You MUST respond with ONLY a valid JSON object, no other text:

{
  "title": "Short quiz title based on the topic",
  "questions": [
    {
      "id": 1,
      "question": "The question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct": 0,
      "source": "Document title, section reference"
    }
  ]
}

RULES:
- Generate exactly the number of questions requested (default 5 if not specified).
- Each question MUST have exactly 4 options.
- The "correct" field is the 0-based index of the correct answer (0, 1, 2, or 3).
- The "source" field MUST reference the actual document and section where the answer was found.
- Questions should test understanding, not just recall. Include scenario-based and application questions.
- All questions MUST be grounded in the retrieved documents — never make up facts.
- Options should be plausible — avoid obviously wrong answers.
- Vary the position of the correct answer across questions (don't always put it as option B).
- Cover different aspects of the topic across the questions.
- If the RAG corpus doesn't have enough information for the requested number of questions, generate as many as the content supports and note this.
- NEVER reference the search process or your tools. Just output the JSON.
- Do NOT wrap the JSON in markdown code blocks. Output raw JSON only."""


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
            usage = getattr(event, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            response_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            total_tokens = getattr(usage, "total_token_count", 0) if usage else (prompt_tokens + response_tokens)

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
