"""POST /api/intent — Intent orchestrator route."""

from fastapi import APIRouter

from config import GCP_PROJECT, GCP_LOCATION, GEN_FAST_MODEL, require_config
from models import IntentRequest, IntentResponse
from agents.intent_orchestrator import run_intent_orchestrator

router = APIRouter()


@router.post("/api/intent", response_model=IntentResponse)
async def intent(req: IntentRequest):
    require_config()
    result = await run_intent_orchestrator(
        req.query,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        model=GEN_FAST_MODEL,
        conversation_history=req.conversationHistory,
    )
    return result
