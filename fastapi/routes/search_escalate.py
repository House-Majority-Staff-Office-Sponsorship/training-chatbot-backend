"""POST /api/search-escalate — Escalation search (Pro) route."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import GCP_PROJECT, GCP_LOCATION, GEN_PRO_MODEL, RAG_CORPUS, require_config
from models import EscalationRequest, SearchResponse
from agents.escalation_search import run_escalation_search

router = APIRouter()


@router.post("/api/search-escalate", response_model=SearchResponse)
async def search_escalate(req: EscalationRequest):
    require_config()
    result = await run_escalation_search(
        req.query,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        model=GEN_PRO_MODEL,
        rag_corpus=RAG_CORPUS,
        previous_answer=req.previousAnswer,
        context=req.context,
        conversation_history=req.conversationHistory,
    )
    return result
