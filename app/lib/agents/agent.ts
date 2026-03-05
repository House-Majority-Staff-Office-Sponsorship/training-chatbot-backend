/**
 * Deep Research Agent — Google ADK multi-squad architecture
 *
 * Inspired by the Google ADK deep-search sample, adapted for RAG-only retrieval.
 *
 * Architecture:
 *
 *   root_agent (interactive_planner_agent — LlmAgent)
 *     ├── plan_generator (AgentTool — LlmAgent)
 *     └── research_pipeline (SequentialAgent)
 *           ├── section_planner (LlmAgent)
 *           ├── research_squad (ParallelAgent)
 *           │     ├── rag_researcher      — broad topic coverage via RAG
 *           │     └── rag_deep_diver      — detailed, specific deep-dives via RAG
 *           ├── iterative_refinement_loop (LoopAgent)
 *           │     ├── research_evaluator  — grades quality, identifies gaps
 *           │     ├── escalation_checker  — breaks loop on "pass"
 *           │     └── hunter_squad (ParallelAgent)
 *           │           ├── rag_gap_filler    — fills identified gaps via RAG
 *           │           └── rag_detail_hunter — hunts for missing details via RAG
 *           └── report_composer (LlmAgent)
 */

import {
  LlmAgent,
  SequentialAgent,
  ParallelAgent,
  LoopAgent,
  AgentTool,
} from "@google/adk";
import { createRagRetrievalTool } from "./tools/rag-tool";
import { EscalationChecker } from "./escalation-checker";

export interface AgentConfig {
  project: string;
  location: string;
  fastModel: string;
  advancedModel: string;
  ragCorpus: string;
  maxRefinementIterations?: number;
}

export function createDeepResearchAgent(config: AgentConfig) {
  const ragTool = createRagRetrievalTool({
    project: config.project,
    location: config.location,
    model: config.advancedModel,
    ragCorpus: config.ragCorpus,
  });

  // ─── Plan Generator (used as AgentTool by the planner) ────────────────────

  const planGenerator = new LlmAgent({
    name: "plan_generator",
    model: config.fastModel,
    description:
      "Generates or refines a 5-point action-oriented research plan.",
    instruction: `You are a research strategist. Create a high-level RESEARCH PLAN (not a summary).
If there is already a plan in session state, improve it based on user feedback.

RESEARCH PLAN (SO FAR):
{research_plan}

Rules:
- Output a bulleted list of exactly 5 action-oriented research goals.
- Each bullet starts with [RESEARCH] for information-gathering tasks or [DELIVERABLE] for synthesis/output tasks.
- A good [RESEARCH] goal starts with a verb: "Analyze," "Identify," "Investigate."
- After the 5 [RESEARCH] goals, add any implied [DELIVERABLE] tasks.
- When refining, mark changes with [MODIFIED] or [NEW].
- Do NOT research the topic yourself — only plan.`,
    outputKey: "research_plan",
  });

  // ─── Section Planner ──────────────────────────────────────────────────────

  const sectionPlanner = new LlmAgent({
    name: "section_planner",
    model: config.fastModel,
    description:
      "Breaks the research plan into a structured markdown outline of report sections.",
    instruction: `You are an expert report architect. Using the research plan from state:

{research_plan}

Design a logical structure for the final report.
- Create a markdown outline with 4-6 distinct sections covering the topic comprehensively.
- Each section should have a heading and brief description of what it covers.
- Do not include a "References" or "Sources" section — citations are handled inline.`,
    outputKey: "report_sections",
  });

  // ─── Research Squad (ParallelAgent) ───────────────────────────────────────

  const ragResearcher = new LlmAgent({
    name: "rag_researcher",
    model: config.advancedModel,
    description:
      "Performs broad research across the RAG corpus following the research plan.",
    instruction: `You are a thorough research agent. Your job is to execute the research plan:

{research_plan}

For each [RESEARCH] goal:
1. Formulate 2-3 targeted queries covering different angles of the goal.
2. Call the retrieve_from_rag tool for EACH query.
3. Synthesize the results into a detailed summary for that goal.

Output all your findings organized by goal. Be comprehensive.`,
    tools: [ragTool],
    outputKey: "broad_research_findings",
  });

  const ragDeepDiver = new LlmAgent({
    name: "rag_deep_diver",
    model: config.advancedModel,
    description:
      "Performs targeted deep-dives into specific topics from the research plan.",
    instruction: `You are a specialist researcher focused on depth over breadth. Review the research plan:

{research_plan}

For each [RESEARCH] goal, identify the single most critical or complex aspect and:
1. Formulate 2-3 highly specific, detailed queries about that aspect.
2. Call the retrieve_from_rag tool for EACH query.
3. Produce an in-depth analysis with specific details, data points, and nuances.

Your findings should complement (not duplicate) broad research by going deeper into specifics.`,
    tools: [ragTool],
    outputKey: "deep_dive_findings",
  });

  const researchSquad = new ParallelAgent({
    name: "research_squad",
    description:
      "Runs broad and deep RAG researchers in parallel for comprehensive coverage.",
    subAgents: [ragResearcher, ragDeepDiver],
  });

  // ─── Research Evaluator ───────────────────────────────────────────────────

  const researchEvaluator = new LlmAgent({
    name: "research_evaluator",
    model: config.advancedModel,
    description:
      "Critically evaluates research quality and identifies gaps.",
    instruction: `You are a meticulous quality assurance analyst. Evaluate the combined research findings:

Broad findings:
{broad_research_findings}

Deep-dive findings:
{deep_dive_findings}

Report structure:
{report_sections}

Evaluate:
1. Comprehensiveness — are all sections of the report adequately covered?
2. Depth — are there sufficient details and specifics?
3. Gaps — what information is missing?
4. Consistency — are there contradictions between findings?

Respond with ONLY a JSON object:
{
  "grade": "pass" or "fail",
  "comment": "detailed explanation of strengths and weaknesses",
  "gaps": ["specific gap 1", "specific gap 2", ...],
  "follow_up_queries": ["targeted query 1", "targeted query 2", ...]
}

Be strict: only grade "pass" if the research thoroughly covers all planned sections.
If grading "fail", provide 3-5 specific follow-up queries to fill the gaps.`,
    outputKey: "research_evaluation",
  });

  // ─── Hunter Squad (ParallelAgent — targets gaps) ──────────────────────────

  const ragGapFiller = new LlmAgent({
    name: "rag_gap_filler",
    model: config.advancedModel,
    description:
      "Fills identified research gaps using targeted RAG queries.",
    instruction: `You are a specialist researcher executing a refinement pass.
The previous research was graded as insufficient. Review the evaluation:

{research_evaluation}

Your job:
1. Parse the "gaps" and "follow_up_queries" from the evaluation.
2. Call retrieve_from_rag for EACH follow-up query.
3. Synthesize new findings that specifically address the identified gaps.
4. Combine your new findings with the existing broad research:

{broad_research_findings}

Output the complete, improved set of broad research findings.`,
    tools: [ragTool],
    outputKey: "broad_research_findings",
  });

  const ragDetailHunter = new LlmAgent({
    name: "rag_detail_hunter",
    model: config.advancedModel,
    description:
      "Hunts for missing details and specifics using targeted RAG queries.",
    instruction: `You are a detail-oriented researcher targeting specific gaps.
The previous research lacked sufficient detail. Review the evaluation:

{research_evaluation}

Your job:
1. Focus on the "gaps" that relate to missing details, data points, or specifics.
2. Formulate 2-3 highly targeted queries for each gap.
3. Call retrieve_from_rag for EACH query.
4. Produce detailed findings that enrich the existing deep-dive research:

{deep_dive_findings}

Output the complete, improved set of deep-dive findings.`,
    tools: [ragTool],
    outputKey: "deep_dive_findings",
  });

  const hunterSquad = new ParallelAgent({
    name: "hunter_squad",
    description:
      "Runs gap-filling and detail-hunting RAG researchers in parallel to address evaluation feedback.",
    subAgents: [ragGapFiller, ragDetailHunter],
  });

  // ─── Iterative Refinement Loop ────────────────────────────────────────────

  const refinementLoop = new LoopAgent({
    name: "iterative_refinement_loop",
    description:
      "Iteratively evaluates research and dispatches the hunter squad until quality passes.",
    maxIterations: config.maxRefinementIterations ?? 2,
    subAgents: [researchEvaluator, new EscalationChecker(), hunterSquad],
  });

  // ─── Report Composer ──────────────────────────────────────────────────────

  const reportComposer = new LlmAgent({
    name: "report_composer",
    model: config.advancedModel,
    description:
      "Transforms research findings into a final, polished report.",
    instruction: `You are an expert report writer. Transform the research into a polished, comprehensive report.

Input data:
- Research Plan: {research_plan}
- Report Structure: {report_sections}
- Broad Research Findings: {broad_research_findings}
- Deep-Dive Findings: {deep_dive_findings}

Instructions:
1. Follow the report structure exactly.
2. Integrate broad and deep-dive findings into each section.
3. Write in clear, professional prose.
4. Be thorough but avoid unnecessary repetition.
5. If findings are contradictory, present both perspectives.
6. Do not reference the research process itself (e.g., don't say "the broad researcher found...").`,
    outputKey: "final_report",
    includeContents: "none",
  });

  // ─── Research Pipeline (SequentialAgent) ──────────────────────────────────

  const researchPipeline = new SequentialAgent({
    name: "research_pipeline",
    description:
      "Executes the full research pipeline: plan sections, parallel research squad, iterative refinement with hunter squad, and final report composition.",
    subAgents: [sectionPlanner, researchSquad, refinementLoop, reportComposer],
  });

  // ─── Root Agent (Interactive Planner) ─────────────────────────────────────

  const rootAgent = new LlmAgent({
    name: "interactive_planner_agent",
    model: config.fastModel,
    description:
      "The primary research assistant. Collaborates with the user to create a research plan, then executes it.",
    instruction: `You are a research planning assistant. Your job is to convert ANY user request into a research plan and then execute it.

Your workflow:
1. PLAN: Use the plan_generator tool to create a draft research plan and present it to the user.
2. REFINE: If the user provides feedback, use plan_generator again to refine the plan.
3. EXECUTE: Once the user approves the plan (e.g., "looks good", "run it", "go ahead"), delegate to the research_pipeline sub-agent to execute the full research.

CRITICAL RULES:
- Never answer a research question directly. Always create a plan first.
- Never perform research yourself. Your only tools are planning and delegation.
- Present the plan clearly with numbered bullet points.
- Ask for user approval before executing.`,
    tools: [new AgentTool({ agent: planGenerator })],
    subAgents: [researchPipeline],
    outputKey: "research_plan",
  });

  return rootAgent;
}
