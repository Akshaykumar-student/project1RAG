from backend.config import Settings


def smtp_settings(**overrides: str | None) -> Settings:
    values = {
        "smtp_host": "smtp.gmail.com",
        "smtp_username": "sender@gmail.com",
        "smtp_password": "valid-google-app-password",
        "smtp_from_email": "sender@gmail.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_email_configuration_requires_complete_smtp_credentials():
    assert smtp_settings().email_configured is True
    assert smtp_settings(smtp_host=None).email_configured is False
    assert smtp_settings(smtp_username=None).email_configured is False
    assert smtp_settings(smtp_password=None).email_configured is False
    assert smtp_settings(smtp_from_email=None).email_configured is False


def test_email_configuration_rejects_placeholder_credentials():
    assert smtp_settings(smtp_username="your_email@gmail.com").email_configured is False
    assert smtp_settings(smtp_password="your_email_app_password").email_configured is False
    assert smtp_settings(smtp_password="replace_with_app_password").email_configured is False
    assert smtp_settings(smtp_password="paste_generated_value_here").email_configured is False
