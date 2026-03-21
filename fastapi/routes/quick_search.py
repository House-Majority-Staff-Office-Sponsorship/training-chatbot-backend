"""POST /api/quick-search — Quick search (Flash) route."""

from fastapi import APIRouter

from config import GCP_PROJECT, GCP_LOCATION, GEN_FAST_MODEL, RAG_CORPUS, require_config
from models import SearchRequest, SearchResponse
from agents.quick_search import run_quick_search

router = APIRouter()


@router.post("/api/quick-search", response_model=SearchResponse)
async def quick_search(req: SearchRequest):
    require_config()
    result = await run_quick_search(
        req.query,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        model=GEN_FAST_MODEL,
        rag_corpus=RAG_CORPUS,
        context=req.context,
        conversation_history=req.conversationHistory,
    )
    return result
