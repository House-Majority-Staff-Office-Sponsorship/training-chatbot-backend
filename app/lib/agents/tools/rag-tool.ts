/**
 * FunctionTool that wraps Vertex AI RAG retrieval.
 *
 * The ADK TypeScript SDK does not ship a built-in VertexAiRagRetrieval tool
 * (that only exists in the Python SDK), so we expose the same capability as a
 * FunctionTool that the LlmAgent can call.
 */

import { FunctionTool } from "@google/adk";
import { VertexAI } from "@google-cloud/vertexai";
import { z } from "zod";

/**
 * Creates a FunctionTool that queries the given Vertex AI RAG corpus.
 *
 * The tool accepts a single `query` string and returns the grounded answer
 * plus any grounding metadata from the RAG engine.
 */
export function createRagRetrievalTool(opts: {
  project: string;
  location: string;
  model: string;
  ragCorpus: string;
}) {
  return new FunctionTool({
    name: "retrieve_from_rag",
    description:
      "Queries the Vertex AI RAG corpus to retrieve grounded information for a given query. " +
      "Call this tool once for EACH sub-query you need to research.",
    parameters: z.object({
      query: z.string().describe("The search query to send to the RAG corpus"),
    }),
    execute: async ({ query }) => {
      const vertexAI = new VertexAI({
        project: opts.project,
        location: opts.location,
      });

      const genModel = vertexAI.getGenerativeModel({
        model: opts.model,
        tools: [
          {
            retrieval: {
              vertexRagStore: {
                ragResources: [{ ragCorpus: opts.ragCorpus }],
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
      const groundingMetadata = candidate?.groundingMetadata ?? null;

      return {
        status: "success",
        query,
        answer,
        groundingMetadata,
      };
    },
  });
}
