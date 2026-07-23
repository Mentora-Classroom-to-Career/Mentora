"""
/interview/* — Phase 2 mocks the RAG pipeline entirely (fixed canned
question + echoed reply). Phase 7 swaps this for real ChromaDB + Ollama
calls, persisting to INTERVIEW_SESSIONS/INTERVIEW_QA.
"""
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from database.models import User
from services.auth_service import get_current_user

router = APIRouter(prefix="/interview", tags=["interview"])


class StartInterviewRequest(BaseModel):
    career_title: str = "Data Scientist"


@router.post("/start")
def start_interview(payload: StartInterviewRequest, current_user: User = Depends(get_current_user)):
    interview_id = str(uuid.uuid4())
    return {
        "interview_id": interview_id,
        "first_question": f"Tell me about a project where you applied your {payload.career_title} skills to solve a real problem.",
    }


class AskInterviewRequest(BaseModel):
    answer: str


@router.post("/{interview_id}/ask")
def ask_interview(interview_id: str, payload: AskInterviewRequest, current_user: User = Depends(get_current_user)):
    # MOCK — deliberately generic so it's never confused with real RAG output later.
    return {"answer": "That's a great point — tell me more about the specific impact you had."}
