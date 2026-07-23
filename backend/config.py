"""
MENTORA backend configuration.

Reads from environment variables (see .env.example). Falls back to sane
local-dev defaults so the server can boot without a .env file present,
but a real deployment MUST set JWT_SECRET explicitly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- Database -----------------------------------------------------
    # Defaults to a local SQLite file so Phase 1/2 work with zero setup.
    # Point DATABASE_URL at Postgres for the "real" master-plan target:
    #   postgresql://postgres:<password>@localhost:5432/mentora_db
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mentora.db")

    # --- Auth / JWT ------------------------------------------------------
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-only-insecure-secret-change-me")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

    # --- CORS ------------------------------------------------------------
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

    # --- Cookie ------------------------------------------------------------
    JWT_COOKIE_NAME: str = os.getenv("JWT_COOKIE_NAME", "mentora_session")

    # --- Misc --------------------------------------------------------------
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi4-mini")


settings = Settings()
