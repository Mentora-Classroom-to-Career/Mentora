from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database.session import engine
from database.models import Base
from routers import auth, users, dashboard, exam, career, learning, interview, notifications

# Phase 1: create tables if they don't exist yet. Once Alembic migrations
# are introduced (not needed for an FYP timeline), this line goes away.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Mentora API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(dashboard.router)
app.include_router(exam.router)
app.include_router(career.router)
app.include_router(learning.router)
app.include_router(interview.router)
app.include_router(notifications.router)


@app.get("/health")
def health():
    return {"status": "ok"}
