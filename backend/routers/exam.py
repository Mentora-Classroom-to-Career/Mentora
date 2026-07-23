"""
/exam/* — Phase 2 mock routers for the /exam-prep page. Field names match
the form `name=` attributes exactly (answer_Q1001-style radios become a
generic `question_id`/`answer` pair once posted as JSON).
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from database.models import User
from services.auth_service import get_current_user

router = APIRouter(prefix="/exam", tags=["exam"])


@router.get("/timetable")
def get_timetable(current_user: User = Depends(get_current_user)):
    return {
        "days_to_exam": 47,
        "rows": [
            {"day": "Monday", "topic": "Algebra — Quadratic Equations", "subject": "Mathematics", "duration": "2 hrs", "status": "pending"},
            {"day": "Tuesday", "topic": "Reading Comprehension", "subject": "English", "duration": "1.5 hrs", "status": "done"},
            {"day": "Wednesday", "topic": "Grammar — Tenses & Voice", "subject": "English", "duration": "2 hrs", "status": "today"},
            {"day": "Thursday", "topic": "Geometry — Lines & Angles", "subject": "Mathematics", "duration": "2 hrs", "status": "missed"},
            {"day": "Friday", "topic": "Revision + MCQ Test", "subject": "Mixed", "duration": "1 hr", "status": "pending"},
        ],
    }


@router.get("/today-topic")
def get_today_topic(current_user: User = Depends(get_current_user)):
    return {
        "topic": "Quadratic Equations",
        "explainer": "A quadratic equation has the form ax\u00b2 + bx + c = 0. Key methods:",
        "key_points": ["Factorization method", "Quadratic formula", "Completing the square"],
        "videos": [
            {"title": "Quadratic Equations — Full Chapter", "source": "MathCity.org", "duration_min": 18},
            {"title": "Quadratic Formula — Step by Step", "source": "Khan Academy", "duration_min": 12},
        ],
    }


@router.get("/mcq")
def get_mcq(topic: str = "quadratic-equations", difficulty: str = "medium", current_user: User = Depends(get_current_user)):
    # MOCK — Phase 6 swaps this for a real M2 (FLAN-T5) generation call.
    return {
        "question_id": "Q1001",
        "question": "Solve x\u00b2 \u2212 5x + 6 = 0. What are the roots?",
        "options": [
            {"value": "A", "label": "x = 2, 3"},
            {"value": "B", "label": "x = \u22122, 3"},
            {"value": "C", "label": "x = 1, 6"},
            {"value": "D", "label": "x = \u22121, \u22126"},
        ],
        "correct_answer": "A",
        "total_in_set": 5,
        "remaining_in_set": 4,
        "generated_by": "FLAN-T5",
    }


class MCQSubmitRequest(BaseModel):
    question_id: str
    answer: str


@router.post("/mcq/submit")
def submit_mcq(payload: MCQSubmitRequest, current_user: User = Depends(get_current_user)):
    # MOCK — Phase 6 swaps this for a real M1 (Gap Classifier) call that
    # only fires on a wrong answer, per the master plan §10.
    correct = payload.answer == "A"
    return {
        "correct": correct,
        "correct_answer": "A",
        "weak_topics_flagged": [] if correct else ["Quadratic Equations"],
    }


class SubjectsUpdateRequest(BaseModel):
    subjects: list[str] | None = None
    test_date: str | None = None
    study_mode: str | None = None


@router.patch("/subjects")
def update_subjects(payload: SubjectsUpdateRequest, current_user: User = Depends(get_current_user)):
    return {"success": True, "subjects": payload.subjects, "test_date": payload.test_date, "study_mode": payload.study_mode}


@router.get("/topic-explainer")
def topic_explainer(query: str, current_user: User = Depends(get_current_user)):
    # MOCK — Phase 7 swaps this for the real ExamRAG (ChromaDB + Ollama) call.
    return {"answer": f"(mock) Here's a grounded explanation related to: {query}"}
