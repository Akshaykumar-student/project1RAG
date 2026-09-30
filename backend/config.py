from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_vector_store_id: str | None = None
    openai_model: str = "gpt-5-mini"
    frontend_origin: str = "http://localhost:5173"
    auth_secret: str = "development-only-change-me"
    auth_database_path: str = "flowdesk.db"
    generated_media_path: str = "generated_media"
    uploaded_attachments_path: str = "uploaded_attachments"
    temporary_uploads_path: str = "temporary_uploads"
    auth_token_minutes: int = 60
    allow_insecure_auth_secret: bool = False
    temporary_upload_cleanup_minutes: int = 15
    max_user_storage_bytes: int = 500 * 1024 * 1024
    auth_rate_limit_per_minute: int = 10
    chat_rate_limit_per_minute: int = 20
    upload_rate_limit_per_minute: int = 12
    frontend_public_url: str = "http://localhost:5173"
    password_reset_minutes: int = 20
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_from_email: str | None = None
    smtp_use_tls: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def configured(self) -> bool:
        return bool(self.openai_api_key and self.vector_store_id)

    def validate_security(self) -> None:
        if self.auth_secret == "development-only-change-me" and not self.allow_insecure_auth_secret:
            raise RuntimeError(
                "AUTH_SECRET is insecure. Set a long random AUTH_SECRET in .env. "
                "For local-only development, set ALLOW_INSECURE_AUTH_SECRET=true."
            )

    @property
    def email_configured(self) -> bool:
        required_values = (
            self.smtp_host,
            self.smtp_username,
            self.smtp_password,
            self.smtp_from_email,
        )
        if not all(value and value.strip() for value in required_values):
            return False

        placeholder_prefixes = ("your_", "replace_", "paste_")
        return all(
            not value.strip().lower().startswith(placeholder_prefixes)
            for value in required_values
            if value
        )

    @property
    def vector_store_id(self) -> str | None:
        if self.openai_vector_store_id:
            return self.openai_vector_store_id
        config_path = Path("vector_store.json")
        if not config_path.exists():
            return None
        import json

        try:
            value = json.loads(config_path.read_text(encoding="utf-8")).get("vector_store_id")
            return value if isinstance(value, str) and value.startswith("vs_") else None
        except (json.JSONDecodeError, OSError):
            return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
