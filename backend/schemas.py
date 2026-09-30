import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def validate_strong_password(value: str) -> str:
    requirements = [
        (len(value) >= 10, "at least 10 characters"),
        (any(character.islower() for character in value), "a lowercase letter"),
        (any(character.isupper() for character in value), "an uppercase letter"),
        (any(character.isdigit() for character in value), "a number"),
        (bool(re.search(r"[^A-Za-z0-9]", value)), "a special character"),
    ]
    missing = [label for valid, label in requirements if not valid]
    if missing:
        raise ValueError("Password must contain " + ", ".join(missing))
    return value


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    name: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        return validate_strong_password(value)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(min_length=1, max_length=128)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=32, max_length=256)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("password")
    @classmethod
    def strong_password(cls, value: str) -> str:
        return validate_strong_password(value)


class MessageResponseBody(BaseModel):
    message: str


class UserResponse(BaseModel):
    id: int
    email: str
    name: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ChatAttachment(BaseModel):
    filename: str = Field(min_length=1, max_length=120)
    mime_type: str = Field(min_length=1, max_length=100)
    encoding: Literal["utf8", "base64"]
    size: int = Field(gt=0, le=10 * 1024 * 1024)
    content: str = Field(min_length=1, max_length=14_000_000)


class ChatRequest(BaseModel):
    board_id: int = Field(gt=0)
    question: str = Field(min_length=2, max_length=1000)
    attachments: list[ChatAttachment] = Field(default_factory=list, max_length=3)
    attachment_ids: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value


class TemporaryUploadResponse(BaseModel):
    id: str
    filename: str
    mime_type: str
    size: int
    preview_type: Literal["pdf", "image", "text", "document"]
    expires_at: int


class Citation(BaseModel):
    label: str
    source_type: Literal["file", "web"] = "file"
    filename: str | None = None
    url: str | None = None


class ChatMedia(BaseModel):
    id: int
    type: Literal["image", "video"]
    mime_type: str
    alt: str


class StoredAttachment(BaseModel):
    id: int
    filename: str
    mime_type: str
    size: int
    preview_type: Literal["pdf", "image", "text", "document"]


class AttachmentPreviewResponse(BaseModel):
    filename: str
    content: str


class MediaError(BaseModel):
    code: Literal[
        "missing_api_key",
        "insufficient_quota",
        "model_access_denied",
        "rate_limit",
        "safety_rejected",
        "generation_timeout",
        "invalid_response",
        "network_error",
        "provider_error",
        "video_unavailable",
    ]
    retryable: bool


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    source: Literal["documentation", "web", "general", "unavailable"]
    media: list[ChatMedia] = Field(default_factory=list)
    media_error: MediaError | None = None


class BoardCreateRequest(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=80)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class BoardRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Title cannot be blank")
        return value.strip()


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    citations: list[Citation]
    source: Literal["user", "documentation", "web", "general", "unavailable"]
    created_at: int
    media: list[ChatMedia] = Field(default_factory=list)
    attachments: list[StoredAttachment] = Field(default_factory=list)
    media_error: MediaError | None = None


class BoardSummaryResponse(BaseModel):
    id: int
    title: str
    created_at: int
    updated_at: int
    message_count: int


class BoardDetailResponse(BaseModel):
    id: int
    title: str
    created_at: int
    updated_at: int
    messages: list[MessageResponse]


class SearchResultResponse(BaseModel):
    board_id: int
    board_title: str
    message_id: int
    role: str
    snippet: str
    created_at: int


class HealthResponse(BaseModel):
    status: str
    configured: bool
    model: str
    attachment_support: bool = True
    missing_attachment_dependencies: list[str] = Field(default_factory=list)


class SampleQuestionsResponse(BaseModel):
    questions: list[str]
