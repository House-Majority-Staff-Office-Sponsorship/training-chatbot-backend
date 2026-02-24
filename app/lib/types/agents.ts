// ---------------------------------------------------------------------------
// Shared types for the deep research agent pipeline
// ---------------------------------------------------------------------------

/**
 * Structured output produced by the IntentExtractor agent.
 * It breaks a single user query into focused sub-queries and extracts key
 * topics so the parallel RAG retrieval step can cast a wider net.
 */
export interface IntentExtractionResult {
  /** The original, unmodified user query. */
  originalQuery: string;
  /**
   * One or more focused sub-queries derived from the original query.
   * The RAG retriever will run each sub-query independently so that
   * different facets of the question are addressed.
   */
  subQueries: string[];
  /** High-level topics / keywords extracted from the query. */
  keyTopics: string[];
  /** Brief restatement of what the user is trying to learn. */
  intent: string;
}

/**
 * The result of a single RAG retrieval + Gemini grounded generation call.
 */
export interface RagRetrievalResult {
  /** The sub-query that produced this result. */
  query: string;
  /** Gemini's grounded answer for this sub-query. */
  answer: string;
  /** Raw grounding / source metadata returned by the Vertex AI RAG engine. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  groundingMetadata: Record<string, any> | null;
}

/**
 * Final output of the deep research pipeline returned to the caller.
 */
export interface DeepResearchResult {
  /** The synthesized, comprehensive answer. */
  answer: string;
  /** The sub-queries that were issued against the RAG corpus. */
  subQueries: string[];
  /** Aggregated grounding metadata from all sub-queries. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  groundingMetadata: (Record<string, any> | null)[];
}
