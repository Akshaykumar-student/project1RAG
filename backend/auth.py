import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, status


@dataclass(frozen=True)
class User:
    id: int
    email: str
    name: str


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return f"pbkdf2_sha256$600000${_encode(salt)}${_encode(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), _decode(salt), int(iterations)
        )
        return hmac.compare_digest(_encode(digest), expected)
    except (TypeError, ValueError):
        return False


class AuthService:
    def __init__(self, database_path: str, secret: str, token_minutes: int = 60) -> None:
        self.database_path = database_path
        self.secret = secret.encode()
        self.token_minutes = token_minutes
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at INTEGER NOT NULL,
                    used_at INTEGER,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_password_reset_expiry "
                "ON password_reset_tokens(expires_at)"
            )

    def register(self, email: str, name: str, password: str) -> User:
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, name, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (email.lower(), name, hash_password(password), int(time.time())),
                )
                return User(id=cursor.lastrowid, email=email.lower(), name=name)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            ) from exc

    def authenticate(self, email: str, password: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, email, name, password_hash FROM users WHERE email = ?",
                (email.lower(),),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return User(id=row["id"], email=row["email"], name=row["name"])

    def create_password_reset(self, email: str, lifetime_minutes: int = 20) -> str | None:
        now = int(time.time())
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM users WHERE email = ?", (email.lower(),)
            ).fetchone()
            connection.execute(
                "DELETE FROM password_reset_tokens WHERE expires_at < ? OR used_at IS NOT NULL",
                (now,),
            )
            if row is None:
                return None
            connection.execute(
                "UPDATE password_reset_tokens SET used_at = ? "
                "WHERE user_id = ? AND used_at IS NULL",
                (now, row["id"]),
            )
            token = secrets.token_urlsafe(48)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            connection.execute(
                """
                INSERT INTO password_reset_tokens
                    (user_id, token_hash, expires_at, used_at, created_at)
                VALUES (?, ?, ?, NULL, ?)
                """,
                (row["id"], token_hash, now + lifetime_minutes * 60, now),
            )
        return token

    def reset_password(self, token: str, password: str) -> bool:
        now = int(time.time())
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id FROM password_reset_tokens
                WHERE token_hash = ? AND used_at IS NULL AND expires_at >= ?
                """,
                (token_hash, now),
            ).fetchone()
            if row is None:
                return False
            connection.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), row["user_id"]),
            )
            connection.execute(
                "UPDATE password_reset_tokens SET used_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            connection.execute(
                "UPDATE password_reset_tokens SET used_at = ? "
                "WHERE user_id = ? AND used_at IS NULL",
                (now, row["user_id"]),
            )
        return True

    def create_token(self, user: User) -> str:
        header = _encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        payload = _encode(
            json.dumps(
                {
                    "sub": str(user.id),
                    "email": user.email,
                    "name": user.name,
                    "exp": int(time.time()) + self.token_minutes * 60,
                },
                separators=(",", ":"),
            ).encode()
        )
        signature = _encode(
            hmac.new(
                self.secret, f"{header}.{payload}".encode(), hashlib.sha256
            ).digest()
        )
        return f"{header}.{payload}.{signature}"

    def user_from_token(self, token: str) -> User | None:
        try:
            header, payload, signature = token.split(".")
            expected = _encode(
                hmac.new(self.secret, f"{header}.{payload}".encode(), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                return None
            claims = json.loads(_decode(payload))
            if int(claims["exp"]) < int(time.time()):
                return None
            user_id = int(claims["sub"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, email, name FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return User(**dict(row)) if row else None
