/**
 * Intent Orchestrator Agent
 *
 * Gatekeeper that validates user queries before they reach the research pipelines.
 * Determines relevance, enriches the query, and signals the frontend to proceed
 * or ask for clarification.
 */

import { LlmAgent, InMemoryRunner } from "@google/adk";
import { type ConversationMessage, formatConversationHistory } from "@/app/lib/types";

export interface IntentOrchestratorConfig {
  project: string;
  location: string;
  model: string;
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

export interface IntentResult {
  action: "confirm" | "clarify" | "reject" | "chat";
  enrichedQuery: string;
  message: string;
  logs: LogEntry[];
}

function createIntentOrchestratorAgent(config: IntentOrchestratorConfig) {
  return new LlmAgent({
    name: "intent_orchestrator",
    model: config.model,
    description:
      "Validates user queries for relevance and enriches them for downstream research agents.",
    instruction: `You are the intent orchestrator for the House Majority Staff Office training chatbot.

PURPOSE OF THIS SYSTEM:
This chatbot helps House Majority staff members — especially new hires — understand internal training documents, policies, procedures, rules, and guidelines. The RAG corpus contains official House Majority training and policy documentation.

YOUR JOB:
Analyze the user's query and determine one of four outcomes:

1. **CONFIRM** — The query is relevant to House Majority training, policies, procedures, or internal documentation. Enrich it and present a brief summary of what you understood so the user can confirm before the research begins.
2. **CHAT** — The query is conversational: a greeting, small talk, a question about what the system can do, or a thank you. No document search needed.
3. **CLARIFY** — The query is ambiguous or too vague to determine relevance. Ask a specific follow-up question to help the user refine their request.
4. **REJECT** — The query is clearly unrelated to House Majority training/policies (e.g., personal questions, general trivia, coding help, unrelated political topics).

DECISION RULES:
- If the query mentions anything about House rules, procedures, training, onboarding, staff policies, ethics, legislative process, committee operations, floor procedures, or any internal House Majority operations → CONFIRM
- If the query is broad but could plausibly relate to training docs (e.g., "tell me about orientation", "what are the rules") → CONFIRM with enrichment
- If the query is conversational (e.g., "hello", "hi", "thanks", "what can you do", "who are you", "good morning") → CHAT
- If the query is too vague to tell (e.g., "help", single words that aren't greetings) → CLARIFY
- If the query is clearly off-topic (e.g., "what's the weather", "write me a poem", "explain quantum physics") → REJECT

WHEN PROCEEDING — Enrich the query:
- ALWAYS frame the enriched query explicitly in the context of the House Majority Staff Office. Every sub-question should reference House Majority training, policies, or procedures.
- Identify the core intent behind the question.
- Infer what kind of training document, policy, or procedure the user is likely asking about.
- Expand abbreviations or shorthand (e.g., "HR" → "House Rules", "CBO" → "Congressional Budget Office").
- Add organizational context: specify which House Majority policies, training modules, or procedural areas are likely relevant.
- Break broad questions into 2-4 specific, searchable sub-questions that are grounded in House Majority operations.
- The enriched query will be passed directly to downstream research agents, so make it detailed and actionable.

OUTPUT FORMAT — You MUST respond with ONLY a JSON object, no other text:

For CONFIRM:
{
  "action": "confirm",
  "enrichedQuery": "Detailed, House Majority-contextualized analysis with specific sub-questions for the research agents",
  "message": "A brief, friendly summary of what you understood the user is asking about (1-3 sentences). This will be shown to the user for confirmation before research begins. End with something like: 'Would you like me to proceed with this search?'"
}

For CHAT:
{
  "action": "chat",
  "enrichedQuery": "",
  "message": ""
}

For CLARIFY:
{
  "action": "clarify",
  "enrichedQuery": "",
  "message": "Your specific follow-up question to the user"
}

For REJECT:
{
  "action": "reject",
  "enrichedQuery": "",
  "message": "A polite explanation that this chatbot is specifically for House Majority training documentation, and suggest what they can ask about instead"
}`,
    outputKey: "intent_result",
  });
}

export async function runIntentOrchestrator(
  query: string,
  config: IntentOrchestratorConfig
): Promise<IntentResult> {
  const agent = createIntentOrchestratorAgent(config);
  const runner = new InMemoryRunner({ agent });

  let rawResult = "";
  const logs: LogEntry[] = [];

  const historyPrefix = formatConversationHistory(config.conversationHistory ?? []);
  const message = historyPrefix + query;

  for await (const event of runner.runEphemeral({
    userId: "intent-user",
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
      logs.push({
        agent: author,
        message: `${author}${detail}`,
        promptTokens,
        responseTokens,
        totalTokens,
        timestamp: Date.now(),
      });
    }

    const delta = event.actions?.stateDelta;
    if (delta && typeof delta["intent_result"] === "string") {
      rawResult = delta["intent_result"];
    }
  }

  // Parse the JSON response from the agent
  try {
    // Extract JSON from the response (agent may wrap it in markdown code blocks)
    const jsonMatch = rawResult.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        action: parsed.action ?? "clarify",
        enrichedQuery: parsed.enrichedQuery ?? "",
        message: parsed.message ?? "",
        logs,
      };
    }
  } catch {
    // If parsing fails, treat as clarify
  }

  return {
    action: "clarify",
    enrichedQuery: "",
    message: "Could you please rephrase your question? I'm here to help with House Majority training documents, policies, and procedures.",
    logs,
  };
}
