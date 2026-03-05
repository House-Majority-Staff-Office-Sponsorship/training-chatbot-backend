/**
 * Programmatic runner for the deep research agent.
 *
 * Uses ADK's InMemoryRunner to execute the multi-squad SequentialAgent pipeline.
 * Supports both batch and streaming modes.
 */

import { InMemoryRunner } from "@google/adk";
import { createDeepResearchAgent, type AgentConfig } from "./agent";
import { formatConversationHistory } from "@/app/lib/types";

export interface DeepResearchResult {
  answer: string;
  enrichedQuery: string;
  researchQuestions: string;
  sectionFindings: string;
}

/** Maps session state keys to their corresponding result field names. */
const STATE_KEY_MAP: Record<string, keyof DeepResearchResult> = {
  enriched_query: "enrichedQuery",
  research_questions: "researchQuestions",
  final_report: "answer",
};

/** A pipeline step event (query analyzer, question expander, report). */
export interface StepEvent {
  type: "step";
  field: keyof DeepResearchResult;
  value: string;
}

/** Emitted when we know how many researchers will run. */
export interface ResearchersInitEvent {
  type: "researchers_init";
  count: number;
  labels: string[];
}

/** Emitted when an individual researcher completes. */
export interface ResearcherDoneEvent {
  type: "researcher_done";
  index: number;
  label: string;
  value: string;
}

/** Log entry emitted for every ADK event with agent name and token usage. */
export interface LogEvent {
  type: "log";
  agent: string;
  message: string;
  promptTokens: number;
  responseTokens: number;
  totalTokens: number;
  timestamp: number;
  researcherIndex?: number;
}

export type PipelineEvent = StepEvent | ResearchersInitEvent | ResearcherDoneEvent | LogEvent;

/**
 * Streaming runner — yields PipelineEvent objects as each pipeline step writes
 * its output to session state.
 */
export async function* streamDeepResearch(
  query: string,
  config: AgentConfig
): AsyncGenerator<PipelineEvent, void, void> {
  const agent = createDeepResearchAgent(config);
  const runner = new InMemoryRunner({ agent });

  const historyPrefix = formatConversationHistory(config.conversationHistory ?? []);
  const message = historyPrefix + query;

  let researcherLabels: string[] = [];

  for await (const event of runner.runEphemeral({
    userId: "research-user",
    newMessage: {
      role: "user",
      parts: [{ text: message }],
    },
  })) {
    // Emit a log event for every ADK event that has an author
    const author = event.author ?? "unknown";
    const usage = event.usageMetadata;
    const promptTokens = usage?.promptTokenCount ?? 0;
    const responseTokens = usage?.candidatesTokenCount ?? 0;
    const totalTokens = usage?.totalTokenCount ?? (promptTokens + responseTokens);

    if (author !== "user") {
      const stateKeys = event.actions?.stateDelta ? Object.keys(event.actions.stateDelta) : [];
      // Skip events whose state deltas are handled specially below (researcher_log, researcher_N, etc.)
      const HANDLED_KEY_RE = /^(researcher_log|researcher_count|researcher_labels|researcher_\d+)$/;
      const unhandledKeys = stateKeys.filter((k) => !HANDLED_KEY_RE.test(k));
      const detail = unhandledKeys.length > 0 ? ` → wrote [${unhandledKeys.join(", ")}]` : "";
      // Skip noise events: no tokens, no meaningful state writes
      if (totalTokens > 0 || unhandledKeys.length > 0) {
        yield {
          type: "log" as const,
          agent: author,
          message: `${author}${detail}`,
          promptTokens,
          responseTokens,
          totalTokens,
          timestamp: Date.now(),
        };
      }
    }

    const delta = event.actions?.stateDelta;
    if (!delta) continue;

    for (const [key, value] of Object.entries(delta)) {
      if (typeof value !== "string") continue;

      // Researcher metadata: labels and count
      if (key === "researcher_labels") {
        try {
          researcherLabels = JSON.parse(value);
        } catch {
          researcherLabels = [];
        }
        const count =
          delta["researcher_count"] != null
            ? Number(delta["researcher_count"])
            : researcherLabels.length;
        yield {
          type: "researchers_init",
          count,
          labels: researcherLabels,
        };
        continue;
      }

      // Skip the count key (already handled above)
      if (key === "researcher_count") continue;

      // Researcher log events from DynamicResearchSquad
      if (key === "researcher_log") {
        try {
          const log = JSON.parse(value);
          yield {
            type: "log" as const,
            agent: log.agent ?? "researcher",
            message: log.message ?? "",
            promptTokens: log.promptTokens ?? 0,
            responseTokens: log.responseTokens ?? 0,
            totalTokens: log.totalTokens ?? 0,
            timestamp: log.timestamp ?? Date.now(),
            researcherIndex: typeof log.researcherIndex === "number" ? log.researcherIndex : undefined,
          };
        } catch {
          // ignore malformed log
        }
        continue;
      }

      // Individual researcher completion: researcher_0, researcher_1, etc.
      const researcherMatch = key.match(/^researcher_(\d+)$/);
      if (researcherMatch) {
        const index = Number(researcherMatch[1]);
        yield {
          type: "researcher_done",
          index,
          label: researcherLabels[index] ?? `Researcher ${index + 1}`,
          value,
        };
        continue;
      }

      // Standard pipeline step
      if (key in STATE_KEY_MAP) {
        yield {
          type: "step",
          field: STATE_KEY_MAP[key],
          value,
        };
      }
    }
  }
}

/**
 * Batch runner — runs the full pipeline and returns collected results.
 */
export async function runDeepResearch(
  query: string,
  config: AgentConfig
): Promise<DeepResearchResult> {
  const result: DeepResearchResult = {
    answer: "(no report produced)",
    enrichedQuery: "",
    researchQuestions: "",
    sectionFindings: "",
  };

  for await (const event of streamDeepResearch(query, config)) {
    if (event.type === "step") {
      result[event.field] = event.value;
    }
  }

  return result;
}
