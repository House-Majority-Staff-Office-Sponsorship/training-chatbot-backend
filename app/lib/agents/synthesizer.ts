/**
 * Synthesizer agent
 *
 * Mirrors the `PatternSynthesizerAgent` / `ReportCompilerAgent` from the ADK
 * blog post.  It receives the individual RAG answers produced by the parallel
 * retrieval step and uses the "advanced" Gemini model to synthesize them into
 * one coherent, comprehensive response.
 */

import { VertexAI } from "@google-cloud/vertexai";
import type { RagRetrievalResult } from "../types/agents";

const SYNTHESIZER_PROMPT = `You are a research synthesis expert. You have been provided with several research findings, each answering a different aspect of the same underlying question. Your job is to synthesize these findings into a single, coherent, and comprehensive answer.

Instructions:
- Integrate all findings without repeating information unnecessarily.
- Present the information in clear, flowing prose.
- If the findings are complementary, combine them naturally.
- If there are contradictions, surface them briefly and explain.
- Be concise but thorough.
- Do not reference the structure of the research process (e.g. do not say "Finding 1 says...").`;

/**
 * Uses the advanced Gemini model to synthesize multiple RAG retrieval results
 * into a single comprehensive answer for the original query.
 */
export async function synthesizeResults(
  originalQuery: string,
  retrievalResults: RagRetrievalResult[],
  options: {
    project: string;
    location: string;
    model: string;
  }
): Promise<string> {
  const vertexAI = new VertexAI({
    project: options.project,
    location: options.location,
  });

  const genModel = vertexAI.getGenerativeModel({ model: options.model });

  // Build a findings block that enumerates each sub-answer
  const findings = retrievalResults
    .map(
      (r, i) =>
        `--- Finding ${i + 1} (sub-query: "${r.query}") ---\n${r.answer}`
    )
    .join("\n\n");

  const prompt = `${SYNTHESIZER_PROMPT}

Original question: ${originalQuery}

Research findings:
${findings}

Please provide a synthesized answer to the original question.`;

  const result = await genModel.generateContent({
    contents: [
      {
        role: "user",
        parts: [{ text: prompt }],
      },
    ],
  });

  return (
    result.response.candidates?.[0]?.content?.parts?.[0]?.text ??
    "(no synthesis available)"
  );
}
