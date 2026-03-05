/**
 * Quick Search Agent — single LlmAgent with RAG tool.
 *
 * Searches the RAG corpus and compiles an answer in a single pass.
 * Much faster than the full deep research pipeline.
 */

import { LlmAgent, InMemoryRunner } from "@google/adk";
import { createRagRetrievalTool, type RagTokenUsage } from "../shared-tools/rag-tool";
import { type ConversationMessage, formatConversationHistory } from "@/app/lib/types";

export interface QuickSearchConfig {
  project: string;
  location: string;
  model: string;
  ragCorpus: string;
  context?: string;
  conversationHistory?: ConversationMessage[];
}

export interface LogEntry {
  agent: string;
  message: string;
  promptTokens: number;
  responseTokens: number;
  totalTokens: number;
  timestamp: number;
}

export interface QuickSearchResult {
  answer: string;
  logs: LogEntry[];
}

function createQuickSearchAgent(config: QuickSearchConfig, onRagTokenUsage?: (usage: RagTokenUsage) => void) {
  const ragTool = createRagRetrievalTool({
    project: config.project,
    location: config.location,
    model: config.model,
    ragCorpus: config.ragCorpus,
    onTokenUsage: onRagTokenUsage,
  });

  return new LlmAgent({
    name: "quick_search_agent",
    model: config.model,
    description:
      "Single-pass research agent that searches and compiles answers from the RAG corpus.",
    instruction: `You are a knowledgeable research assistant for the House Majority Staff Office. You help staff members — especially new hires — understand internal training documents, policies, procedures, and guidelines by searching the official document corpus.

Your job:
1. Analyze the user's question and identify 2-4 targeted search queries that cover different angles.
2. You MUST call the retrieve_from_rag tool for EACH query. Always search — never refuse or say you cannot.
3. Synthesize ALL retrieved information into a single, clear, comprehensive answer.

If the user's question is broad or vague, break it down into specific sub-topics and search for each one. Never ask the user to rephrase or provide more detail — always make your best effort with what they gave you.

Rules:
- You MUST always call the retrieve_from_rag tool at least once. Never respond without searching first.
- Be thorough but concise — aim for a well-structured response, not a lengthy report.
- If the corpus doesn't contain relevant information for a query, say so honestly in your answer.
- NEVER reference the search process, your tools, your capabilities, or your limitations. Do not say things like "I searched for...", "The RAG returned...", "My function only allows...", or "I would need to know...". Just answer the question directly.
- NEVER tell the user to provide more specific queries or rephrase their question.
- Present information in clear prose with bullet points or sections where appropriate.
- Always back up your answer with evidence: cite policy numbers, section references, and document titles.
- Quote key definitions, rules, and requirements verbatim using quotation marks.
- Include specific data points, dates, and thresholds exactly as they appear in the source.
- End your answer with a "## Sources" section listing every document title, policy number, and URI from the RAG responses.`,
    tools: [ragTool],
    outputKey: "quick_answer",
  });
}

export async function runQuickSearch(
  query: string,
  config: QuickSearchConfig
): Promise<QuickSearchResult> {
  const ragLogs: LogEntry[] = [];
  const agent = createQuickSearchAgent(config, (usage) => {
    ragLogs.push({
      agent: "retrieve_from_rag",
      message: `retrieve_from_rag → "${usage.query.length > 60 ? usage.query.slice(0, 57) + "..." : usage.query}"`,
      promptTokens: usage.promptTokens,
      responseTokens: usage.responseTokens,
      totalTokens: usage.totalTokens,
      timestamp: usage.timestamp,
    });
  });
  const runner = new InMemoryRunner({ agent });

  const historyPrefix = formatConversationHistory(config.conversationHistory ?? []);
  // If intent orchestrator provided enriched context, prepend it
  const message = historyPrefix + (config.context
    ? `INTENT ANALYSIS (use this to guide your search — all queries relate to House Majority Staff Office):\n${config.context}\n\nUSER QUESTION:\n${query}`
    : query);

  let answer = "";
  const logs: LogEntry[] = [];

  for await (const event of runner.runEphemeral({
    userId: "quick-search-user",
    newMessage: {
      role: "user",
      parts: [{ text: message }],
    },
  })) {
    const author = event.author ?? "unknown";
    if (author !== "user") {
      const usage = event.usageMetadata;
      const promptTokens = usage?.promptTokenCount ?? 0;
      const responseTokens = usage?.candidatesTokenCount ?? 0;
      const totalTokens = usage?.totalTokenCount ?? (promptTokens + responseTokens);
      const stateKeys = event.actions?.stateDelta ? Object.keys(event.actions.stateDelta) : [];
      const detail = stateKeys.length > 0 ? ` → wrote [${stateKeys.join(", ")}]` : "";
      // Skip noise events (e.g. "tool response") that have no tokens and no state writes
      if (totalTokens > 0 || stateKeys.length > 0) {
        logs.push({
          agent: author,
          message: `${author}${detail}`,
          promptTokens,
          responseTokens,
          totalTokens,
          timestamp: Date.now(),
        });
      }
    }

    const delta = event.actions?.stateDelta;
    if (delta && typeof delta["quick_answer"] === "string") {
      answer = delta["quick_answer"];
    }
  }

  // Merge RAG tool token logs into the timeline
  const allLogs = [...logs, ...ragLogs].sort((a, b) => a.timestamp - b.timestamp);

  return { answer: answer || "(no answer produced)", logs: allLogs };
}
