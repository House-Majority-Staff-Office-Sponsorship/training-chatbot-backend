/**
 * IntentExtractor agent
 *
 * Mirrors the `intent_extractor_agent` from the ADK blog post.
 * Uses the "fast" Gemini model to analyse the user's raw query and produce
 * a structured `IntentExtractionResult` — specifically a list of focused
 * sub-queries that will each be sent to the RAG retriever.
 */

import { VertexAI } from "@google-cloud/vertexai";
import type { IntentExtractionResult } from "../types/agents";

const INTENT_EXTRACTOR_PROMPT = `You are an expert query analyst. Your job is to decompose a user question into focused sub-queries so that a retrieval system can answer each part thoroughly.

Given the user query, respond with ONLY a valid JSON object (no markdown fences, no extra text) with this exact shape:
{
  "originalQuery": "<the exact user query>",
  "subQueries": ["<sub-query 1>", "<sub-query 2>", ...],
  "keyTopics": ["<topic 1>", "<topic 2>", ...],
  "intent": "<one-sentence restatement of what the user wants to learn>"
}

Rules:
- Produce 2–4 sub-queries that together cover all aspects of the original question.
- Each sub-query must be a complete, self-contained question.
- Keep keyTopics concise (1–3 words each).
- Do NOT include any text outside the JSON object.`;

/**
 * Calls the Gemini "fast" model to decompose `query` into structured intent.
 */
export async function extractIntent(
  query: string,
  options: {
    project: string;
    location: string;
    model: string;
  }
): Promise<IntentExtractionResult> {
  const vertexAI = new VertexAI({
    project: options.project,
    location: options.location,
  });

  const genModel = vertexAI.getGenerativeModel({ model: options.model });

  const result = await genModel.generateContent({
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `${INTENT_EXTRACTOR_PROMPT}\n\nUser query: ${query}`,
          },
        ],
      },
    ],
  });

  const raw =
    result.response.candidates?.[0]?.content?.parts?.[0]?.text ?? "";

  // Strip optional markdown code fences that some model responses include
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();

  let parsed: IntentExtractionResult;
  try {
    parsed = JSON.parse(cleaned) as IntentExtractionResult;
  } catch {
    // Fallback: treat the entire query as a single sub-query so the pipeline
    // can still proceed even if the model did not return valid JSON.
    parsed = {
      originalQuery: query,
      subQueries: [query],
      keyTopics: [],
      intent: query,
    };
  }

  // Always guarantee the original query is preserved
  parsed.originalQuery = query;

  // Ensure at least one sub-query exists
  if (!Array.isArray(parsed.subQueries) || parsed.subQueries.length === 0) {
    parsed.subQueries = [query];
  }

  return parsed;
}
