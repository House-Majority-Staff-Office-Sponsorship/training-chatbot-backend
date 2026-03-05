/**
 * Shared types used across all agents and API routes.
 */

/** A single turn in the conversation history. */
export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * Formats conversation history into a text block that can be prepended
 * to an agent's user message for context.
 */
export function formatConversationHistory(
  history: ConversationMessage[]
): string {
  if (history.length === 0) return "";
  const lines = history.map(
    (m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`
  );
  return `CONVERSATION HISTORY (use this for context — the user may refer to earlier messages):\n${lines.join("\n")}\n\n`;
}
