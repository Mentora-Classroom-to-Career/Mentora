from fastapi import APIRouter, Depends

from database.models import User
from services.auth_service import get_current_user

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("")
def get_notifications(current_user: User = Depends(get_current_user)):
    return {
        "items": [
            {"id": 1, "message": "Your weak topic 'Quadratic Equations' has new practice questions.", "is_read": False},
            {"id": 2, "message": "Career ranking updated — Data Scientist is now your top match.", "is_read": False},
            {"id": 3, "message": "Reminder: Wednesday's Grammar session is due today.", "is_read": True},
        ]
    }
