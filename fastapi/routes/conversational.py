"""POST /api/conversational — Conversational chat route."""

from fastapi import APIRouter

from config import GCP_PROJECT, GCP_LOCATION, GEN_FAST_MODEL, require_config
from models import SearchRequest, SearchResponse
from agents.conversational import run_conversational

router = APIRouter()


@router.post("/api/conversational", response_model=SearchResponse)
async def conversational(req: SearchRequest):
    require_config()
    result = await run_conversational(
        req.query,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        model=GEN_FAST_MODEL,
        conversation_history=req.conversationHistory,
    )
    return result
