/**
 * RagRetriever agent
 *
 * Mirrors the per-company research step from the ADK blog post.
 * For a single sub-query it calls Vertex AI with the RAG corpus attached so
 * Gemini performs grounded retrieval and returns a cited answer.
 */

import { VertexAI } from "@google-cloud/vertexai";
import type { RagRetrievalResult } from "../types/agents";

/**
 * Runs a single grounded-retrieval call against the RAG corpus for `query`.
 */
export async function retrieveFromRag(
  query: string,
  options: {
    project: string;
    location: string;
    model: string;
    ragCorpus: string;
  }
): Promise<RagRetrievalResult> {
  const vertexAI = new VertexAI({
    project: options.project,
    location: options.location,
  });

  const genModel = vertexAI.getGenerativeModel({
    model: options.model,
    tools: [
      {
        retrieval: {
          vertexRagStore: {
            ragResources: [{ ragCorpus: options.ragCorpus }],
          },
        },
      },
    ],
  });

  const result = await genModel.generateContent({
    contents: [
      {
        role: "user",
        parts: [{ text: query }],
      },
    ],
  });

  const candidate = result.response.candidates?.[0];
  const answer = candidate?.content?.parts?.[0]?.text ?? "(no response)";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const groundingMetadata = (candidate?.groundingMetadata as Record<string, any> | null | undefined) ?? null;

  return { query, answer, groundingMetadata };
}
