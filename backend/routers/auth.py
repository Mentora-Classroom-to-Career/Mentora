"""
/auth/* — the only routes that stay open (no get_current_user dependency),
per master plan §7.2. Everything else in the app requires a valid JWT.
"""
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.session import get_db
from database.models import User, StudentIntelligenceProfile
from schemas import (
    RegisterRequest, LoginRequest, AuthResponse, UserPublic,
    ForgotPasswordRequest, ResetPasswordRequest, ChangePasswordRequest, SimpleSuccess,
)
from services.auth_service import (
    hash_password, verify_password, create_access_token, generate_reset_code,
    get_current_user,
)

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("mentora.auth")

# Fixed exam_goal values from the register form's <select> — kept here so
# a typo'd value fails loudly instead of silently landing in the DB.
VALID_EXAM_GOALS = {
    "sindh_university", "mehran_university", "lums", "karachi_university",
    "iqra_university", "tando_jam_university", "pms_spsc_ppsc", "issb", "army_navy_paf",
}


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    if payload.exam_goal not in VALID_EXAM_GOALS:
        raise HTTPException(status_code=422, detail=f"Invalid exam_goal: {payload.exam_goal}")

    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        university=payload.university,
        exam_goal=payload.exam_goal,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Seed the Student Intelligence Profile — every feature reads/writes
    # this record, so it must exist from the moment of registration.
    sip = StudentIntelligenceProfile(user_id=user.id, skill_vector={}, topic_scores_summary={}, priority_queue=[])
    db.add(sip)
    db.commit()

    token = create_access_token(user.id)
    return AuthResponse(access_token=token, user=UserPublic.model_validate(user))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user.id)
    return AuthResponse(access_token=token, user=UserPublic.model_validate(user))


@router.post("/forgot-password", response_model=SimpleSuccess)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    # Always return success even if the email doesn't exist — don't leak
    # which emails are registered.
    if user:
        code = generate_reset_code()
        user.reset_code = code
        user.reset_code_expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
        db.commit()
        # Phase 1 has no email service wired up yet — log it so the flow
        # is testable locally. Wire a real mailer (or Ollama-free SMTP)
        # before this ever leaves localhost.
        logger.info(f"[DEV] Password reset code for {user.email}: {code}")
    return SimpleSuccess()


@router.post("/reset-password", response_model=SimpleSuccess)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=422, detail="Passwords do not match")

    user = db.query(User).filter(User.reset_code == payload.reset_code).first()
    if not user or not user.reset_code_expires_at:
        raise HTTPException(status_code=400, detail="Invalid or expired reset code")
    if datetime.now(timezone.utc) > user.reset_code_expires_at.replace(tzinfo=timezone.utc):
        raise HTTPException(status_code=400, detail="Reset code has expired")

    user.password_hash = hash_password(payload.new_password)
    user.reset_code = None
    user.reset_code_expires_at = None
    db.commit()
    return SimpleSuccess()


@router.post("/change-password", response_model=SimpleSuccess)
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    current_user.password_hash = hash_password(payload.new_password)
    db.commit()
    return SimpleSuccess()
