"""
FunctionTool that wraps Vertex AI RAG retrieval.

Uses `rag.retrieval_query` to return RAW retrieved chunks (not an
LLM-synthesized summary). The LLM agent then parses embedded structured
fields — page numbers, section IDs, policy identifiers — directly from
the JSONL chunk text.

Supports top_k and hybrid search alpha configuration.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from google.adk.tools import FunctionTool


class RagTokenUsage:
    def __init__(self, query: str, prompt_tokens: int, response_tokens: int, total_tokens: int, timestamp: int):
        self.query = query
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.total_tokens = total_tokens
        self.timestamp = timestamp


def create_rag_retrieval_tool(
    *,
    project: str,
    location: str,
    model: str,
    rag_corpus: str,
    top_k: int = 10,
    hybrid_search_alpha: float = 0.5,
    on_token_usage: Optional[Callable[[RagTokenUsage], None]] = None,
) -> FunctionTool:
    """
    Creates a FunctionTool that queries the Vertex AI RAG corpus and returns
    the raw retrieved chunks for the agent to parse and cite.

    Args:
        top_k: Number of top chunks to retrieve (default: 10).
        hybrid_search_alpha: Balance between semantic (0.0) and keyword (1.0) search.
            0.5 = equal weight (default). 0.0 = pure semantic. 1.0 = pure keyword.

    Note:
        `model` is accepted for signature compatibility but no longer used —
        this tool now performs pure retrieval with no nested LLM call.
    """

    del model  # retained in signature for backward compatibility with callers

    def retrieve_from_rag(query: str) -> dict:
        """Queries the Vertex AI RAG corpus and returns the top retrieved chunks verbatim.
        Each chunk includes its raw text (scan for embedded page numbers, section IDs,
        policy identifiers), source_uri, source_display_name, and relevance score.
        Call this tool once for EACH sub-query you need to research."""

        import vertexai as vtx
        from vertexai.preview import rag

        vtx.init(project=project, location=location)

        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=top_k,
            hybrid_search=rag.HybridSearch(alpha=hybrid_search_alpha),
        )

        response = rag.retrieval_query(
            text=query,
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
            rag_retrieval_config=rag_retrieval_config,
        )

        # Extract contexts — response.contexts.contexts per the SDK shape
        contexts = []
        raw_contexts = getattr(response, "contexts", None)
        if raw_contexts is not None:
            inner = getattr(raw_contexts, "contexts", None)
            contexts = list(inner) if inner is not None else list(raw_contexts)

        # Format each chunk as a clearly-delimited block so the LLM can parse
        # embedded structured fields (page, section, policy_id, etc.) from the
        # chunk text itself.
        blocks: list[str] = []
        for i, ctx in enumerate(contexts, start=1):
            text = getattr(ctx, "text", "") or ""
            source_uri = getattr(ctx, "source_uri", "") or ""
            display = getattr(ctx, "source_display_name", "") or ""
            score = getattr(ctx, "score", None)
            distance = getattr(ctx, "distance", None)

            header_parts = [f"[Chunk {i}]"]
            if score is not None:
                header_parts.append(f"score={score:.3f}")
            elif distance is not None:
                header_parts.append(f"distance={distance:.3f}")
            if display:
                header_parts.append(f"file={display}")
            if source_uri and source_uri != display:
                header_parts.append(f"uri={source_uri}")

            blocks.append(" | ".join(header_parts) + "\n" + text.strip())

        if blocks:
            formatted = "\n\n---\n\n".join(blocks)
        else:
            formatted = "(no chunks retrieved for this query)"

        # No nested LLM call — report 0 tokens but keep callback shape
        # so timeline logging continues to function.
        if on_token_usage:
            on_token_usage(RagTokenUsage(
                query=query,
                prompt_tokens=0,
                response_tokens=0,
                total_tokens=0,
                timestamp=int(time.time() * 1000),
            ))

        return {
            "status": "success",
            "query": query,
            "answer": formatted,
            "chunkCount": len(contexts),
            "ragTokens": {
                "prompt": 0,
                "response": 0,
                "total": 0,
            },
        }

    return FunctionTool(func=retrieve_from_rag)
