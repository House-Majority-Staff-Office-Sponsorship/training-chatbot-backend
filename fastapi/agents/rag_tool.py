"""
FunctionTool that wraps Vertex AI RAG retrieval.

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
    Creates a FunctionTool that queries the Vertex AI RAG corpus.

    Args:
        top_k: Number of top chunks to retrieve (default: 10).
        hybrid_search_alpha: Balance between semantic (0.0) and keyword (1.0) search.
            0.5 = equal weight (default). 0.0 = pure semantic. 1.0 = pure keyword.
    """

    def retrieve_from_rag(query: str) -> dict:
        """Queries the Vertex AI RAG corpus to retrieve grounded information for a given query.
        Call this tool once for EACH sub-query you need to research."""

        import vertexai as vtx
        from vertexai.generative_models import GenerativeModel, Tool as VertexTool, Content, Part
        from vertexai.preview import rag

        vtx.init(project=project, location=location)

        # Configure RAG retrieval with top_k and hybrid search
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=top_k,
            hybrid_search=rag.HybridSearch(alpha=hybrid_search_alpha),
        )

        rag_resource = rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
            rag_retrieval_config=rag_retrieval_config,
        )

        retrieval = rag.Retrieval(source=rag_resource)

        gen_model = GenerativeModel(
            model_name=model,
            tools=[VertexTool.from_retrieval(retrieval=retrieval)],
        )

        response = gen_model.generate_content(
            contents=[Content(role="user", parts=[Part.from_text(query)])]
        )

        candidate = response.candidates[0] if response.candidates else None
        answer = candidate.content.parts[0].text if candidate and candidate.content.parts else "(no response)"

        # Extract token usage
        usage = response.usage_metadata
        rag_prompt_tokens = usage.prompt_token_count if usage else 0
        rag_response_tokens = usage.candidates_token_count if usage else 0
        rag_total_tokens = usage.total_token_count if usage else (rag_prompt_tokens + rag_response_tokens)

        # Extract source references from grounding metadata
        sources: list[str] = []
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        if grounding_metadata:
            chunks = getattr(grounding_metadata, "grounding_chunks", []) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                retrieved = getattr(chunk, "retrieved_context", None)
                title = ""
                uri = ""
                if web:
                    title = getattr(web, "title", "") or ""
                    uri = getattr(web, "uri", "") or ""
                elif retrieved:
                    title = getattr(retrieved, "title", "") or ""
                    uri = getattr(retrieved, "uri", "") or ""
                if title or uri:
                    sources.append(f"{title} ({uri})" if title and uri else (title or uri))

        sources_text = ""
        if sources:
            source_lines = [f"[{i + 1}] {s}" for i, s in enumerate(sources)]
            sources_text = "\n\nSOURCES:\n" + "\n".join(source_lines)

        # Notify caller of RAG token usage
        if on_token_usage:
            on_token_usage(RagTokenUsage(
                query=query,
                prompt_tokens=rag_prompt_tokens,
                response_tokens=rag_response_tokens,
                total_tokens=rag_total_tokens,
                timestamp=int(time.time() * 1000),
            ))

        return {
            "status": "success",
            "query": query,
            "answer": answer + sources_text,
            "sourceCount": len(sources),
            "ragTokens": {
                "prompt": rag_prompt_tokens,
                "response": rag_response_tokens,
                "total": rag_total_tokens,
            },
        }

    return FunctionTool(func=retrieve_from_rag)
