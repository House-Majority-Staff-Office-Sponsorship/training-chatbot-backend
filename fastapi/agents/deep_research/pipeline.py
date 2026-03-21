"""
Deep Research Agent — Google ADK multi-squad architecture

Architecture:
  deep_research_pipeline (SequentialAgent)
    ├── query_analyzer          — enriches raw user query
    ├── question_expander       — breaks topic into sub-questions
    ├── dynamic_research_squad  — spawns one researcher per sub-question, all in parallel
    └── report_compiler         — compiles all research into final report
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, SequentialAgent

from agents.deep_research.dynamic_research_squad import DynamicResearchSquad


def create_deep_research_agent(
    *,
    fast_model: str,
    advanced_model: str,
    report_model: str,
    project: str,
    location: str,
    rag_corpus: str,
) -> SequentialAgent:

    # ── Query Analyzer ───────────────────────────────────────────────────
    query_analyzer = LlmAgent(
        name="query_analyzer",
        model=fast_model,
        description="Analyzes and enriches the user's raw query with organizational context for more effective research.",
        instruction="""You are a query analyst for the House Majority Staff Office training documentation system.

Your users are House Majority staff members — often new hires — who need help understanding internal training documents, policies, procedures, rules, and guidelines.

Your job is to take the user's raw question and produce an ENRICHED QUERY that will drive the rest of the research pipeline more effectively.

Steps:
1. Identify the core intent behind the question.
2. Infer what kind of training document, policy, or procedure the user is likely asking about.
3. Expand abbreviations or shorthand the user may have used.
4. Add relevant context: what specific policies, rules, procedures, or training topics might be involved.
5. Break a broad or vague question into 2-4 specific sub-questions that should be answered.

Output format:
- **Original Query**: (repeat the user's question)
- **Intent**: (one sentence describing what the user really needs to know)
- **Context**: (what area of House Majority training/policy this relates to)
- **Enriched Sub-Questions**: (2-4 specific, searchable questions that together fully answer the original query)

Do NOT answer the question — only analyze and enrich it for downstream agents.""",
        output_key="enriched_query",
    )

    # ── Question Expander ────────────────────────────────────────────────
    question_expander = LlmAgent(
        name="question_expander",
        model=advanced_model,
        description="Expands the enriched query into exactly 5 detailed sub-questions for parallel research.",
        instruction="""You receive a query analysis and produce exactly 5 research sub-questions. No more, no fewer.

Query analysis: {enriched_query}

Rules:
- Output EXACTLY 5 questions. Not 4, not 6. Exactly 5.
- Each question must end with a question mark.
- One question per line, numbered 1-5.
- No extra text, no preamble, no explanations, no headings, no markdown formatting.

Example output:
1. What is the policy on X?
2. Are there exceptions to rule Y?
3. How does procedure Z apply to new hires?
4. What are the deadlines for completing W?
5. Who is responsible for enforcing V?""",
        output_key="research_questions",
    )

    # ── Dynamic Research Squad ───────────────────────────────────────────
    research_squad = DynamicResearchSquad(
        project=project,
        location=location,
        model=advanced_model,
        rag_corpus=rag_corpus,
    )

    # ── Report Compiler ──────────────────────────────────────────────────
    report_compiler = LlmAgent(
        name="research_compiler",
        model=report_model,
        description="Compiles all research findings and sources into a single organized output.",
        instruction="""You compile research findings for House Majority staff.

Input:
- Query: {enriched_query}
- Research questions: {research_questions}
- Findings: {section_findings_all}

Output format:
1. Start with a brief executive summary answering the user's original question.
2. Then organize the findings by topic. Keep all citations, quotes, and source references exactly as they appear in the findings.
3. End with a "## Sources" section listing every document title, policy number, and URI from the findings.

Do not add a memo header. Do not reference the research process.""",
        output_key="final_report",
    )

    # ── Full Pipeline ────────────────────────────────────────────────────
    return SequentialAgent(
        name="deep_research_pipeline",
        description="Full auto-executing deep research pipeline: query analysis, question expansion, dynamic per-question parallel research, and final report composition.",
        sub_agents=[
            query_analyzer,
            question_expander,
            research_squad,
            report_compiler,
        ],
    )
