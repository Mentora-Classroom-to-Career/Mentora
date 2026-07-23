"""
/career/* — Phase 2 mock routers for the /career page. cv-upload accepts
the real multipart file (field name `cv_file`, matching the form) but
returns fixed mock extraction data — no NER model runs yet (that's
Phase 6, M5).
"""
from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel

from database.models import User
from services.auth_service import get_current_user

router = APIRouter(prefix="/career", tags=["career"])

_CAREER_RANKINGS = [
    {"career_id": 1, "title": "Data Scientist", "match_score": 88, "skill_gap": "Deep Learning, TensorFlow"},
    {"career_id": 2, "title": "ML Engineer", "match_score": 78, "skill_gap": "TensorFlow, MLOps"},
    {"career_id": 3, "title": "Business Analyst", "match_score": 66, "skill_gap": "SQL, Tableau, Excel"},
    {"career_id": 4, "title": "Software Engineer", "match_score": 58, "skill_gap": "System Design, APIs"},
    {"career_id": 5, "title": "Product Manager", "match_score": 42, "skill_gap": "Leadership, Strategy, UX"},
]

_ROADMAP_STEPS = [
    {"step": "1-2", "title": "Python & Stats", "subtitle": "Fundamentals", "status": "completed"},
    {"step": "3", "title": "ML Algorithms", "subtitle": "Scikit-learn", "status": "completed"},
    {"step": "4", "title": "Deep Learning", "subtitle": "TensorFlow", "status": "in_progress"},
    {"step": "5", "title": "Projects & RAG", "subtitle": "Portfolio", "status": "pending"},
    {"step": "6", "title": "Mock Interviews", "subtitle": "Job Applications", "status": "pending"},
]


@router.post("/cv-upload")
async def upload_cv(cv_file: UploadFile, current_user: User = Depends(get_current_user)):
    # MOCK — Phase 6 swaps this for a real M5 (BERT NER) extraction call.
    _ = await cv_file.read()  # confirm the multipart body actually arrives
    return {
        "filename": cv_file.filename,
        "gpa": 3.6,
        "technical_skills": ["Python", "Data Analysis", "SQL"],
        "certifications": ["AWS Cloud Found."],
        "projects": ["Sentiment Analyzer"],
    }


class CareerGoalRequest(BaseModel):
    career_goal: str


@router.post("/goal")
def set_career_goal(payload: CareerGoalRequest, current_user: User = Depends(get_current_user)):
    return {"success": True, "career_goal": payload.career_goal}


@router.get("/ai-suggest")
def ai_suggest(current_user: User = Depends(get_current_user)):
    # MOCK — Phase 6 swaps this for a real M4 (Sentence-Transformers) call.
    top = _CAREER_RANKINGS[0]
    return {"suggested_career_id": top["career_id"], "suggested_title": top["title"], "match_score": top["match_score"]}


@router.get("/rankings")
def get_rankings(current_user: User = Depends(get_current_user)):
    return {"rankings": _CAREER_RANKINGS}


@router.get("/roadmap")
def get_roadmap(career_id: int = 1, current_user: User = Depends(get_current_user)):
    career = next((c for c in _CAREER_RANKINGS if c["career_id"] == career_id), _CAREER_RANKINGS[0])
    return {"career_id": career["career_id"], "career_title": career["title"], "steps": _ROADMAP_STEPS}
