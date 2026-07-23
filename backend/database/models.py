"""
SQLAlchemy models, built directly from the MENTORA ERD
(MENTORA_Complete_Project_Plan.md §4 / §10).

Field names on USERS match the `name=` attributes in the real Next.js
register/login forms exactly (first_name, last_name, email, password,
university, exam_goal) so Phase 2's request bodies need no translation
layer between frontend and DB.

Only Phase 1 (auth) actually writes/reads USERS + a seeded
STUDENT_INTELLIGENCE_PROFILE today. The remaining tables are modeled now
so Phase 2's mock routers and Phase 6's real ones have a stable schema to
target later — none of them are queried for real data until Phase 6+.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    university = Column(String(255), nullable=True)
    # one of the fixed exam_goal <select> values from the register form
    exam_goal = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=_now)

    # password-reset flow (Phase 1 §4.4 of the frontend doc)
    reset_code = Column(String(6), nullable=True)
    reset_code_expires_at = Column(DateTime, nullable=True)

    sip = relationship("StudentIntelligenceProfile", back_populates="user", uselist=False)
    exam_sessions = relationship("ExamSession", back_populates="user")
    career_rankings = relationship("CareerRanking", back_populates="user")
    cv_uploads = relationship("CVUpload", back_populates="user")
    interview_sessions = relationship("InterviewSession", back_populates="user")
    notifications = relationship("Notification", back_populates="user")


class StudentIntelligenceProfile(Base):
    """1:1 with USERS — one record per student holding the SIP."""
    __tablename__ = "student_intelligence_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    skill_vector = Column(JSON, nullable=True)      # embedding / skill-weight dict
    topic_scores_summary = Column(JSON, nullable=True)
    priority_queue = Column(JSON, nullable=True)     # ordered list of topics to study next
    career_goal = Column(String(255), nullable=True)
    roadmap_id = Column(Integer, ForeignKey("roadmaps.id"), nullable=True)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    user = relationship("User", back_populates="sip")


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_text = Column(Text, nullable=False)
    subject = Column(String(100), nullable=False)     # e.g. "Mathematics" — must match frontend subject chips
    topic = Column(String(150), nullable=False)
    subtopic = Column(String(150), nullable=True)
    difficulty = Column(String(20), nullable=True)     # easy | medium | hard
    option_a = Column(String(500), nullable=False)
    option_b = Column(String(500), nullable=False)
    option_c = Column(String(500), nullable=False)
    option_d = Column(String(500), nullable=False)
    correct_answer = Column(String(1), nullable=False)  # "A" | "B" | "C" | "D"
    source = Column(String(50), default="bank")           # "bank" | "generated" (M2 output)

    answers = relationship("SessionAnswer", back_populates="question")


class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_number = Column(Integer, nullable=False)
    subject = Column(String(100), nullable=True)
    score = Column(Float, nullable=True)
    started_at = Column(DateTime, default=_now)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="exam_sessions")
    answers = relationship("SessionAnswer", back_populates="session")


class SessionAnswer(Base):
    __tablename__ = "session_answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    selected_answer = Column(String(1), nullable=True)
    is_correct = Column(Boolean, nullable=True)
    answered_at = Column(DateTime, default=_now)

    session = relationship("ExamSession", back_populates="answers")
    question = relationship("Question", back_populates="answers")


class TopicScore(Base):
    __tablename__ = "topic_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject = Column(String(100), nullable=False)
    topic = Column(String(150), nullable=False)
    score = Column(Float, nullable=False)
    note = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=_now, onupdate=_now)


class CareerProfile(Base):
    __tablename__ = "career_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    required_skills = Column(JSON, nullable=True)  # list[str]

    rankings = relationship("CareerRanking", back_populates="career")
    roadmaps = relationship("Roadmap", back_populates="career")


class CareerRanking(Base):
    __tablename__ = "career_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    career_id = Column(Integer, ForeignKey("career_profiles.id"), nullable=False)
    match_score = Column(Float, nullable=False)   # 0-100, from M4 cosine similarity
    rank = Column(Integer, nullable=True)
    computed_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="career_rankings")
    career = relationship("CareerProfile", back_populates="rankings")


class Roadmap(Base):
    __tablename__ = "roadmaps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    career_id = Column(Integer, ForeignKey("career_profiles.id"), nullable=False)
    steps = Column(JSON, nullable=True)  # list[{month, title, description}]

    career = relationship("CareerProfile", back_populates="roadmaps")


class CVUpload(Base):
    __tablename__ = "cv_uploads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    extracted_skills = Column(JSON, nullable=True)
    extracted_certs = Column(JSON, nullable=True)
    extracted_gpa = Column(Float, nullable=True)
    uploaded_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="cv_uploads")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    career_title = Column(String(255), nullable=False)
    started_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="interview_sessions")
    qa_turns = relationship("InterviewQA", back_populates="session")


class InterviewQA(Base):
    __tablename__ = "interview_qa"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("interview_sessions.id"), nullable=False)
    question = Column(Text, nullable=False)
    student_answer = Column(Text, nullable=True)
    ai_reply = Column(Text, nullable=True)
    turn_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=_now)

    session = relationship("InterviewSession", back_populates="qa_turns")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(String(500), nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="notifications")
