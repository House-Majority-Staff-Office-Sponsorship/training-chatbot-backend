/**
 * Deep Research Orchestrator
 *
 * Inspired by the SequentialAgent + parallel sub-agent pattern described in
 * the Google ADK blog post.  The pipeline runs three stages in order:
 *
 *   1. IntentExtractor  – decomposes the user query into focused sub-queries.
 *   2. RagRetriever     – queries the RAG corpus for EACH sub-query in
 *                         parallel (mirrors the ResearchOrchestratorAgent).
 *   3. Synthesizer      – merges all retrieval results into one coherent
 *                         answer (mirrors the PatternSynthesizerAgent).
 *
 * A single call to `runDeepResearch()` drives the entire pipeline and returns
 * the final `DeepResearchResult`.
 */

import { extractIntent } from "./intent-extractor";
import { retrieveFromRag } from "./rag-retriever";
import { synthesizeResults } from "./synthesizer";
import type { DeepResearchResult } from "../types/agents";

export interface OrchestratorOptions {
  /** GCP project ID */
  project: string;
  /** Vertex AI region */
  location: string;
  /** Fast Gemini model used for intent extraction */
  fastModel: string;
  /** Advanced Gemini model used for synthesis */
  advancedModel: string;
  /** Full Vertex AI RAG corpus resource name */
  ragCorpus: string;
}

/**
 * Runs the full deep research pipeline for a single user query.
 *
 * Stage 1 – Intent extraction (fast model, single call)
 * Stage 2 – Parallel RAG retrieval  (advanced model, one call per sub-query)
 * Stage 3 – Synthesis  (advanced model, single call)
 */
export async function runDeepResearch(
  query: string,
  opts: OrchestratorOptions
): Promise<DeepResearchResult> {
  // ── Stage 1: Intent Extraction ────────────────────────────────────────────
  const intent = await extractIntent(query, {
    project: opts.project,
    location: opts.location,
    model: opts.fastModel,
  });

  // ── Stage 2: Parallel RAG Retrieval ───────────────────────────────────────
  // Each sub-query is dispatched concurrently (mirrors the parallel
  // ResearchOrchestratorAgent in the ADK architecture).
  const retrievalResults = await Promise.all(
    intent.subQueries.map((subQuery) =>
      retrieveFromRag(subQuery, {
        project: opts.project,
        location: opts.location,
        model: opts.advancedModel,
        ragCorpus: opts.ragCorpus,
      })
    )
  );

  // ── Stage 3: Synthesis ────────────────────────────────────────────────────
  const synthesizedAnswer = await synthesizeResults(
    query,
    retrievalResults,
    {
      project: opts.project,
      location: opts.location,
      model: opts.advancedModel,
    }
  );

  return {
    answer: synthesizedAnswer,
    subQueries: intent.subQueries,
    groundingMetadata: retrievalResults.map((r) => r.groundingMetadata),
  };
}
