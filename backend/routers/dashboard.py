"""
/dashboard/* — Phase 2: every response below is realistic, hardcoded mock
JSON, shaped exactly like the real future response (same field names,
same nesting, numbers as numbers). Swapped for real ModelManager/DB
queries in Phase 6 — see MENTORA_Phase6_Model_Manager_Inference.md.
"""
from fastapi import APIRouter, Depends

from database.models import User
from services.auth_service import get_current_user

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
def get_stats(current_user: User = Depends(get_current_user)):
    return {
        "exam_sessions": 24,
        "avg_score": 72,
        "avg_score_delta": 5,
        "videos_watched": 38,
        "videos_watched_this_week": 14,
        "top_career_match": {"title": "Data Scientist", "fit_percent": 88},
    }


@router.get("/score-trend")
def get_score_trend(current_user: User = Depends(get_current_user), sessions: int = 10):
    return {
        "points": [
            {"session": f"S{i+1}", "score": v}
            for i, v in enumerate([55, 60, 68, 65, 75, 72, 82, 80, 90, 88])
        ]
    }


@router.get("/weekly-snapshot")
def get_weekly_snapshot(current_user: User = Depends(get_current_user)):
    return {
        "days": [
            {"day": "Mon", "topic": "Algebra", "status": "done"},
            {"day": "Tue", "topic": "English", "status": "done"},
            {"day": "Wed", "topic": "Grammar", "status": "today"},
            {"day": "Thu", "topic": "Geometry", "status": "missed"},
            {"day": "Fri", "topic": "Revision", "status": "pending"},
        ]
    }


@router.get("/weak-topics")
def get_weak_topics(current_user: User = Depends(get_current_user)):
    return {
        "topics": [
            {"subject": "Algebra", "topic": "Quadratic Equations", "score": 38, "note": "Needs urgent review"},
            {"subject": "Grammar", "topic": "Tenses", "score": 44, "note": None},
            {"subject": "Geometry", "topic": "Circles", "score": 60, "note": "Medium"},
            {"subject": "English", "topic": "Reading Comprehension", "score": 62, "note": "Improving"},
        ]
    }



