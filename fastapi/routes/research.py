"""POST /api/research — Deep research SSE streaming route."""

from __future__ import annotations

import json
import traceback

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import GCP_PROJECT, GCP_LOCATION, GEN_FAST_MODEL, GEN_REPORT_MODEL, RAG_CORPUS, require_config
from models import SearchRequest
from agents.deep_research.runner import stream_deep_research

router = APIRouter()


@router.post("/api/research")
async def research(req: SearchRequest):
    require_config()

    # If intent orchestrator provided enriched context, prepend it
    enriched_query = req.query
    if req.context:
        enriched_query = (
            f"INTENT ANALYSIS (use this to guide your research — all queries relate to House Majority Staff Office):\n"
            f"{req.context}\n\nUSER QUESTION:\n{req.query}"
        )

    async def event_generator():
        try:
            async for event in stream_deep_research(
                enriched_query,
                project=GCP_PROJECT,
                location=GCP_LOCATION,
                fast_model=GEN_FAST_MODEL,
                advanced_model=GEN_REPORT_MODEL,
                report_model=GEN_REPORT_MODEL,
                rag_corpus=RAG_CORPUS,
                conversation_history=req.conversationHistory,
            ):
                event_type = event["type"]

                if event_type == "log":
                    payload = {
                        "agent": event["agent"],
                        "message": event["message"],
                        "promptTokens": event["promptTokens"],
                        "responseTokens": event["responseTokens"],
                        "totalTokens": event["totalTokens"],
                        "timestamp": event["timestamp"],
                    }
                    if event.get("researcherIndex") is not None:
                        payload["researcherIndex"] = event["researcherIndex"]
                    yield f"event: log\ndata: {json.dumps(payload)}\n\n"

                elif event_type == "step":
                    payload = {"field": event["field"], "value": event["value"]}
                    yield f"event: step\ndata: {json.dumps(payload)}\n\n"

                elif event_type == "researchers_init":
                    payload = {"count": event["count"], "labels": event["labels"]}
                    yield f"event: researchers_init\ndata: {json.dumps(payload)}\n\n"

                elif event_type == "researcher_done":
                    payload = {"index": event["index"], "label": event["label"], "value": event["value"]}
                    yield f"event: researcher_done\ndata: {json.dumps(payload)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            error_msg = str(e)
            print(f"[/api/research] SSE pipeline error: {error_msg}")
            traceback.print_exc()
            payload = json.dumps({"error": "Pipeline failed.", "detail": error_msg})
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
