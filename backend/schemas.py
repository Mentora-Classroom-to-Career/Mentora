"""
Pydantic schemas. Field names mirror the `name=` attributes in the real
Next.js forms exactly (see MENTORA_Phase1_Frontend_Documentation.md §4,
§8) so no translation layer is needed between frontend and API.
"""
from pydantic import BaseModel, EmailStr, field_validator


# ---------------------------------------------------------------- auth --

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    confirm_password: str
    university: str
    exam_goal: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info):
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserPublic"


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    reset_code: str
    new_password: str
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class SimpleSuccess(BaseModel):
    success: bool = True


# --------------------------------------------------------------- users --

class UserPublic(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: EmailStr
    university: str | None = None
    exam_goal: str | None = None

    class Config:
        from_attributes = True


class UserUpdateRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    university: str | None = None
    exam_goal: str | None = None


AuthResponse.model_rebuild()
