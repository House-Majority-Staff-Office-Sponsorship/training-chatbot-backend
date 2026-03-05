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
export interface RagTokenUsage {
  query: string;
  promptTokens: number;
  responseTokens: number;
  totalTokens: number;
  timestamp: number;
}

export function createRagRetrievalTool(opts: {
  project: string;
  location: string;
  model: string;
  ragCorpus: string;
  onTokenUsage?: (usage: RagTokenUsage) => void;
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

      // Extract token usage from the Vertex AI response
      const usageMeta = result.response.usageMetadata;
      const ragPromptTokens = usageMeta?.promptTokenCount ?? 0;
      const ragResponseTokens = usageMeta?.candidatesTokenCount ?? 0;
      const ragTotalTokens = usageMeta?.totalTokenCount ?? (ragPromptTokens + ragResponseTokens);

      // Extract source references from grounding metadata
      const sources: string[] = [];
      if (groundingMetadata) {
        const chunks =
          (groundingMetadata as Record<string, unknown>).groundingChunks ??
          (groundingMetadata as Record<string, unknown>).retrievedContext ??
          [];
        if (Array.isArray(chunks)) {
          for (const chunk of chunks) {
            const c = chunk as Record<string, unknown>;
            const web = c.web as Record<string, unknown> | undefined;
            const retrievedContext = c.retrievedContext as Record<string, unknown> | undefined;
            const title =
              (web?.title as string) ??
              (retrievedContext?.title as string) ??
              (c.title as string) ??
              "";
            const uri =
              (web?.uri as string) ??
              (retrievedContext?.uri as string) ??
              (c.uri as string) ??
              "";
            if (title || uri) {
              sources.push(title ? `${title}${uri ? ` (${uri})` : ""}` : uri);
            }
          }
        }
      }

      const sourcesText =
        sources.length > 0
          ? `\n\nSOURCES:\n${sources.map((s, i) => `[${i + 1}] ${s}`).join("\n")}`
          : "";

      // Notify caller of RAG token usage
      opts.onTokenUsage?.({
        query,
        promptTokens: ragPromptTokens,
        responseTokens: ragResponseTokens,
        totalTokens: ragTotalTokens,
        timestamp: Date.now(),
      });

      return {
        status: "success",
        query,
        answer: answer + sourcesText,
        sourceCount: sources.length,
        ragTokens: { prompt: ragPromptTokens, response: ragResponseTokens, total: ragTotalTokens },
      };
    },
  });
}
