/**
 * Programmatic runner for the deep research agent.
 *
 * Uses ADK's InMemoryRunner to execute the multi-squad SequentialAgent pipeline
 * and collect the final report from session state.
 */

import { InMemoryRunner } from "@google/adk";
import { createDeepResearchAgent, type AgentConfig } from "./agent";

export interface DeepResearchResult {
  answer: string;
  researchPlan: string;
  reportSections: string;
  broadFindings: string;
  deepDiveFindings: string;
  evaluation: string;
}

/**
 * Runs the full deep research pipeline for a single user query using
 * ADK's InMemoryRunner.
 *
 * The runner creates an ephemeral session, executes the agent hierarchy
 * (planner → research squad → refinement loop w/ hunter squad → report),
 * and returns the collected results.
 */
export async function runDeepResearch(
  query: string,
  config: AgentConfig
): Promise<DeepResearchResult> {
  const agent = createDeepResearchAgent(config);
  const runner = new InMemoryRunner({ agent });

  const state: Record<string, string> = {};

  for await (const event of runner.runEphemeral({
    userId: "research-user",
    newMessage: {
      role: "user",
      parts: [{ text: query }],
    },
  })) {
    const delta = event.actions?.stateDelta;
    if (delta) {
      for (const [key, value] of Object.entries(delta)) {
        if (typeof value === "string") {
          state[key] = value;
        }
      }
    }
  }

  return {
    answer: state["final_report"] || "(no report produced)",
    researchPlan: state["research_plan"] || "",
    reportSections: state["report_sections"] || "",
    broadFindings: state["broad_research_findings"] || "",
    deepDiveFindings: state["deep_dive_findings"] || "",
    evaluation: state["research_evaluation"] || "",
  };
}
