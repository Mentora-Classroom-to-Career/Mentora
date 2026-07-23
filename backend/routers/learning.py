from fastapi import APIRouter, Depends

from database.models import User
from services.auth_service import get_current_user

router = APIRouter(prefix="/learning", tags=["learning"])


@router.get("/stats")
def get_stats(current_user: User = Depends(get_current_user)):
    return {
        "videos_in_playlist": 38,
        "videos_watched_this_week": 14,
        "mcq_assignments": 12,
        "mcq_pending_today": 3,
        "course_pdfs": 7,
        "course_pdfs_downloaded": 2,
        "avg_mcq_score": 72,
        "avg_mcq_score_delta": 5,
    }


@router.get("/playlist")
def get_playlist(current_user: User = Depends(get_current_user)):
    return {
        "videos": [
            {"id": 1, "title": "Quadratic Equations — Full Chapter", "source": "MathCity.org", "duration_min": 18, "subject": "Mathematics", "watched": True},
            {"id": 2, "title": "Quadratic Formula — Step by Step", "source": "Khan Academy", "duration_min": 12, "subject": "Mathematics", "watched": False},
            {"id": 3, "title": "Reading Comprehension Strategies", "source": "English Hub", "duration_min": 15, "subject": "English", "watched": True},
        ]
    }


@router.get("/assignments")
def get_assignments(current_user: User = Depends(get_current_user)):
    return {
        "assignments": [
            {"id": 1, "title": "Algebra Practice Set 4", "subject": "Mathematics", "status": "pending", "due": "Today"},
            {"id": 2, "title": "Grammar — Tenses Quiz", "subject": "English", "status": "pending", "due": "Tomorrow"},
            {"id": 3, "title": "Geometry Basics Review", "subject": "Mathematics", "status": "done", "due": "Yesterday"},
        ]
    }


@router.get("/materials")
def get_materials(current_user: User = Depends(get_current_user)):
    return {
        "materials": [
            {"id": 1, "title": "Sindh University Entry Test — Math Formula Sheet", "subject": "Mathematics", "downloaded": True},
            {"id": 2, "title": "English Grammar Handbook", "subject": "English", "downloaded": False},
        ]
    }


@router.get("/materials/{material_id}/download")
def download_material(material_id: int, current_user: User = Depends(get_current_user)):
    # MOCK — Phase 6 streams a real file; for now just confirm the route works.
    return {"material_id": material_id, "download_url": f"/static/materials/{material_id}.pdf"}
