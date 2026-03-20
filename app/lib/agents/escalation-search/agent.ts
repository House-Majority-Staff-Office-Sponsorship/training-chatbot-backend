/**
 * Escalation Search Agent — Pro model with RAG tool.
 *
 * Activated when the user indicates the initial Flash quick-search answer
 * was not satisfactory. Re-searches the RAG corpus with the Pro model,
 * taking the previous answer into account to go deeper and cover gaps.
 */

import { LlmAgent, InMemoryRunner } from "@google/adk";
import { createRagRetrievalTool, type RagTokenUsage } from "../shared-tools/rag-tool";
import { type ConversationMessage, formatConversationHistory } from "@/app/lib/types";

export interface EscalationSearchConfig {
  project: string;
  location: string;
  model: string;
  ragCorpus: string;
  context?: string;
  previousAnswer: string;
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

export interface EscalationSearchResult {
  answer: string;
  logs: LogEntry[];
}

function createEscalationSearchAgent(
  config: EscalationSearchConfig,
  onRagTokenUsage?: (usage: RagTokenUsage) => void
) {
  const ragTool = createRagRetrievalTool({
    project: config.project,
    location: config.location,
    model: config.model,
    ragCorpus: config.ragCorpus,
    onTokenUsage: onRagTokenUsage,
  });

  return new LlmAgent({
    name: "escalation_search_agent",
    model: config.model,
    description:
      "Deep follow-up research agent that re-searches the RAG corpus with greater thoroughness after an initial answer was insufficient.",
    instruction: `You are an advanced research assistant for the House Majority Staff Office. A previous search attempt produced an answer that the user found insufficient. Your job is to conduct a MORE THOROUGH search and provide a significantly better answer.

The previous answer will be provided in the user message. Analyze it to understand what was already covered and what gaps remain.

Your approach:
1. Analyze WHY the previous answer may have been insufficient — it may have been too shallow, missed key details, covered the wrong angle, or lacked specifics.
2. Formulate 4-6 targeted search queries that go DEEPER than the previous attempt. Try different angles, more specific terminology, and related policy areas that the first search may have missed.
3. You MUST call the retrieve_from_rag tool for EACH query. Always search — never refuse or say you cannot.
4. Synthesize ALL retrieved information into a comprehensive, well-structured answer that clearly improves upon the previous one.

Rules:
- You MUST always call the retrieve_from_rag tool at least once. Never respond without searching first.
- Be thorough and comprehensive — the user already got a quick answer and wants more depth.
- If the corpus doesn't contain relevant information for a query, say so honestly in your answer.
- NEVER reference the search process, your tools, your capabilities, or your limitations. Do not say things like "I searched for...", "The RAG returned...", "My function only allows...", or "I would need to know...". Just answer the question directly.
- NEVER reference the previous answer or say things like "Building on the previous answer..." or "The earlier response missed...". Just provide a complete, standalone answer.
- NEVER tell the user to provide more specific queries or rephrase their question.
- Present information in clear prose with bullet points or sections where appropriate.
- Always back up your answer with evidence: cite policy numbers, section references, and document titles.
- Quote key definitions, rules, and requirements verbatim using quotation marks.
- Include specific data points, dates, and thresholds exactly as they appear in the source.
- End your answer with a "## Sources" section listing every document title, policy number, and URI from the RAG responses.`,
    tools: [ragTool],
    outputKey: "escalation_answer",
  });
}

export async function runEscalationSearch(
  query: string,
  config: EscalationSearchConfig
): Promise<EscalationSearchResult> {
  const ragLogs: LogEntry[] = [];
  const agent = createEscalationSearchAgent(config, (usage) => {
    ragLogs.push({
      agent: "retrieve_from_rag",
      message: `retrieve_from_rag → "${usage.query.length > 60 ? usage.query.slice(0, 57) + "..." : usage.query}"`,
      promptTokens: usage.promptTokens,
      responseTokens: usage.responseTokens,
      totalTokens: usage.totalTokens,
      timestamp: usage.timestamp,
    });
  });

  // Inject the previous answer into the agent's instruction via state
  const runner = new InMemoryRunner({ agent });

  const historyPrefix = formatConversationHistory(config.conversationHistory ?? []);
  // Build the message with context, including the previous answer for the agent to analyze
  let message = historyPrefix;
  if (config.context) {
    message += `INTENT ANALYSIS (use this to guide your search — all queries relate to House Majority Staff Office):\n${config.context}\n\n`;
  }
  message += `PREVIOUS ANSWER (the user was NOT satisfied with this):\n---\n${config.previousAnswer}\n---\n\nUSER QUESTION:\n${query}`;

  let answer = "";
  const logs: LogEntry[] = [];

  for await (const event of runner.runEphemeral({
    userId: "escalation-search-user",
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
    if (delta && typeof delta["escalation_answer"] === "string") {
      answer = delta["escalation_answer"];
    }
  }

  // Merge RAG tool token logs into the timeline
  const allLogs = [...logs, ...ragLogs].sort((a, b) => a.timestamp - b.timestamp);

  return { answer: answer || "(no answer produced)", logs: allLogs };
}
