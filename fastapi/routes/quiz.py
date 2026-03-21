"""POST /api/quiz/generate — Generate a quiz from RAG corpus content."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from config import GCP_PROJECT, GCP_LOCATION, GEN_FAST_MODEL, RAG_CORPUS, require_config
from agents.quiz_generator import generate_quiz

router = APIRouter()


class QuizGenerateRequest(BaseModel):
    topic: str
    numQuestions: Optional[int] = 5


@router.post("/api/quiz/generate")
async def quiz_generate(req: QuizGenerateRequest):
    require_config()
    result = await generate_quiz(
        req.topic,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        model=GEN_FAST_MODEL,
        rag_corpus=RAG_CORPUS,
        num_questions=req.numQuestions or 5,
    )
    return result
