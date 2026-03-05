/**
 * Conversational Agent — simple LLM chat with no tools.
 *
 * Handles greetings, small talk, and general questions about
 * what the system can do. No RAG search, no research pipeline.
 */

import { LlmAgent, InMemoryRunner } from "@google/adk";
import { type ConversationMessage, formatConversationHistory } from "@/app/lib/types";

export interface ConversationalConfig {
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

export interface ConversationalResult {
  answer: string;
  logs: LogEntry[];
}

function createConversationalAgent(config: ConversationalConfig) {
  return new LlmAgent({
    name: "conversational_agent",
    model: config.model,
    description:
      "Handles conversational queries — greetings, small talk, and questions about the system.",
    instruction: `You are a friendly assistant for the House Majority Staff Office training chatbot.

You handle conversational messages — greetings, small talk, and questions about what this system can do.

About this system:
- This chatbot helps House Majority staff members understand internal training documents, policies, procedures, and guidelines.
- Users can ask questions about House rules, onboarding, staff policies, ethics, legislative process, committee operations, floor procedures, and other internal House Majority operations.
- The system searches an official document corpus to provide grounded, sourced answers.
- There are three search modes: Quick Search (fast single-pass), Quick Search Pro (higher quality), and Deep Research (thorough multi-agent pipeline).

Rules:
- Be friendly, helpful, and concise.
- If someone greets you, greet them back warmly and briefly explain what you can help with.
- If someone asks what you can do, explain the system's capabilities.
- If someone asks a question that sounds like it needs document research, suggest they try one of the search modes.
- Keep responses short — 2-4 sentences for greetings, a bit more if explaining capabilities.
- Do not make up information about House Majority policies or procedures.`,
    outputKey: "chat_answer",
  });
}

export async function runConversational(
  query: string,
  config: ConversationalConfig
): Promise<ConversationalResult> {
  const agent = createConversationalAgent(config);
  const runner = new InMemoryRunner({ agent });

  let answer = "";
  const logs: LogEntry[] = [];

  const historyPrefix = formatConversationHistory(config.conversationHistory ?? []);
  const message = historyPrefix + query;

  for await (const event of runner.runEphemeral({
    userId: "chat-user",
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
    if (delta && typeof delta["chat_answer"] === "string") {
      answer = delta["chat_answer"];
    }
  }

  return { answer: answer || "(no response)", logs };
}
