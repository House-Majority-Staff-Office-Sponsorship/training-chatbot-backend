"""POST /api/quick-search-pro — Quick search (Pro) route."""

from fastapi import APIRouter

from config import GCP_PROJECT, GCP_LOCATION, GEN_PRO_MODEL, RAG_CORPUS, require_config
from models import SearchRequest, SearchResponse
from agents.quick_search import run_quick_search

router = APIRouter()


@router.post("/api/quick-search-pro", response_model=SearchResponse)
async def quick_search_pro(req: SearchRequest):
    require_config()
    result = await run_quick_search(
        req.query,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        model=GEN_PRO_MODEL,
        rag_corpus=RAG_CORPUS,
        context=req.context,
        conversation_history=req.conversationHistory,
    )
    return result
