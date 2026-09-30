from pathlib import Path

import pytest

from backend.auth import AuthService, hash_password, verify_password


def test_password_hash_is_salted_and_verifiable():
    first = hash_password("correct-horse-battery-staple")
    second = hash_password("correct-horse-battery-staple")
    assert first != second
    assert verify_password("correct-horse-battery-staple", first)
    assert not verify_password("wrong-password", first)


def test_register_authenticate_and_token_round_trip(tmp_path: Path):
    auth = AuthService(str(tmp_path / "users.db"), "test-secret", token_minutes=5)
    user = auth.register("USER@example.com", "Test User", "long-password")

    assert user.email == "user@example.com"
    assert auth.authenticate("user@example.com", "long-password") == user
    assert auth.authenticate("user@example.com", "bad-password") is None
    assert auth.user_from_token(auth.create_token(user)) == user


def test_duplicate_registration_is_rejected(tmp_path: Path):
    auth = AuthService(str(tmp_path / "users.db"), "test-secret")
    auth.register("user@example.com", "First", "long-password")
    with pytest.raises(Exception) as error:
        auth.register("USER@example.com", "Second", "long-password")
    assert getattr(error.value, "status_code", None) == 409


def test_tampered_token_is_rejected(tmp_path: Path):
    auth = AuthService(str(tmp_path / "users.db"), "test-secret")
    user = auth.register("user@example.com", "Test", "long-password")
    token = auth.create_token(user)
    assert auth.user_from_token(token + "tampered") is None


def test_password_reset_token_is_single_use(tmp_path: Path):
    auth = AuthService(str(tmp_path / "auth.db"), "test-secret")
    auth.register("user@example.com", "Test User", "OldPassword1")
    token = auth.create_password_reset("user@example.com", lifetime_minutes=20)

    assert token is not None
    assert auth.reset_password(token, "NewPassword2") is True
    assert auth.authenticate("user@example.com", "NewPassword2") is not None
    assert auth.authenticate("user@example.com", "OldPassword1") is None
    assert auth.reset_password(token, "AnotherPassword3") is False


def test_password_reset_does_not_reveal_unknown_email(tmp_path: Path):
    auth = AuthService(str(tmp_path / "auth.db"), "test-secret")

    assert auth.create_password_reset("missing@example.com") is None
