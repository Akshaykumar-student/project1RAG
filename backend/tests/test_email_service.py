from email.message import EmailMessage

import pytest

from backend.config import Settings
from backend.email_service import EmailConfigurationError, send_password_reset_email

pytestmark = pytest.mark.asyncio


def smtp_settings(**overrides) -> Settings:
    values = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_username": "sender@example.com",
        "smtp_password": "app-password",
        "smtp_from_email": "sender@example.com",
        "smtp_use_tls": True,
        "password_reset_minutes": 20,
    }
    values.update(overrides)
    return Settings(**values)


async def test_password_reset_email_requires_complete_configuration():
    config = Settings(
        smtp_host=None,
        smtp_username=None,
        smtp_password=None,
        smtp_from_email=None,
    )
    with pytest.raises(EmailConfigurationError, match="not fully configured"):
        await send_password_reset_email(config, "user@example.com", "https://reset.test")


async def test_password_reset_email_uses_tls_login_and_expected_message(monkeypatch):
    events: list[object] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            events.append(("connect", host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("close")

        def starttls(self):
            events.append("tls")

        def login(self, username, password):
            events.append(("login", username, password))

        def send_message(self, message: EmailMessage):
            events.append(("message", message))

    monkeypatch.setattr("backend.email_service.smtplib.SMTP", FakeSMTP)
    config = smtp_settings()

    await send_password_reset_email(
        config, "user@example.com", "http://localhost:5173?reset_token=secret"
    )

    assert events[:3] == [
        ("connect", "smtp.example.com", 587, 20),
        "tls",
        ("login", "sender@example.com", "app-password"),
    ]
    message = next(
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == "message"
    )
    assert message["Subject"] == "Reset your FlowDesk password"
    assert message["From"] == "sender@example.com"
    assert message["To"] == "user@example.com"
    assert "reset_token=secret" in message.get_content()
    assert "20 minutes" in message.get_content()
    assert events[-1] == "close"


async def test_password_reset_email_can_skip_starttls(monkeypatch):
    tls_started = False

    class FakeSMTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def starttls(self):
            nonlocal tls_started
            tls_started = True

        def login(self, *_):
            pass

        def send_message(self, *_):
            pass

    monkeypatch.setattr("backend.email_service.smtplib.SMTP", FakeSMTP)

    await send_password_reset_email(
        smtp_settings(smtp_use_tls=False), "user@example.com", "https://reset.test"
    )

    assert tls_started is False
