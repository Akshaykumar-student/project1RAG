import asyncio
import smtplib
from email.message import EmailMessage

from backend.config import Settings


class EmailConfigurationError(RuntimeError):
    pass


async def send_password_reset_email(config: Settings, recipient: str, reset_url: str) -> None:
    if not config.email_configured:
        raise EmailConfigurationError(
            "Password-reset email is not fully configured. Add a valid SMTP username, "
            "app password, host, and sender address to .env."
        )

    message = EmailMessage()
    message["Subject"] = "Reset your FlowDesk password"
    message["From"] = config.smtp_from_email
    message["To"] = recipient
    message.set_content(
        "We received a request to reset your FlowDesk password.\n\n"
        f"Open this link to choose a new password:\n{reset_url}\n\n"
        f"This link expires in {config.password_reset_minutes} minutes and can be used once.\n"
        "If you did not request this, you can ignore this email."
    )

    def send() -> None:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=20) as smtp:
            if config.smtp_use_tls:
                smtp.starttls()
            if config.smtp_username:
                smtp.login(config.smtp_username, config.smtp_password or "")
            smtp.send_message(message)

    await asyncio.to_thread(send)
