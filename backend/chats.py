import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException, status


class ChatStore:
    def __init__(
        self,
        database_path: str,
        media_path: str = "generated_media",
        attachment_path: str = "uploaded_attachments",
        temporary_upload_path: str = "temporary_uploads",
    ) -> None:
        self.database_path = database_path
        self.media_path = Path(media_path)
        self.attachment_path = Path(attachment_path)
        self.temporary_upload_path = Path(temporary_upload_path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_boards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    board_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'general',
                    media_error TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (board_id) REFERENCES chat_boards(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_media (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    media_type TEXT NOT NULL CHECK (media_type IN ('image', 'video')),
                    mime_type TEXT NOT NULL,
                    filename TEXT NOT NULL UNIQUE,
                    alt TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL UNIQUE,
                    mime_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    preview_type TEXT NOT NULL,
                    preview_text TEXT,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY (message_id) REFERENCES chat_messages(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS temporary_uploads (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL UNIQUE,
                    mime_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    preview_type TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_boards_user_updated
                    ON chat_boards(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_board_created
                    ON chat_messages(board_id, created_at, id);
                CREATE INDEX IF NOT EXISTS idx_media_message
                    ON chat_media(message_id, id);
                CREATE INDEX IF NOT EXISTS idx_attachments_message
                    ON chat_attachments(message_id, id);
                CREATE INDEX IF NOT EXISTS idx_temporary_uploads_expiry
                    ON temporary_uploads(expires_at);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_messages)").fetchall()
            }
            if "source" not in columns:
                connection.execute(
                    "ALTER TABLE chat_messages ADD COLUMN source TEXT NOT NULL DEFAULT 'general'"
                )
                connection.execute(
                    """
                    UPDATE chat_messages
                    SET source = CASE
                        WHEN role = 'user' THEN 'user'
                        WHEN citations <> '[]' THEN 'documentation'
                        WHEN content = ? THEN 'unavailable'
                        ELSE 'general'
                    END
                    """,
                    (
                        "I don't have enough information in the FlowDesk documentation "
                        "to answer that question.",
                    ),
                )
            if "media_error" not in columns:
                connection.execute("ALTER TABLE chat_messages ADD COLUMN media_error TEXT")

    def list_boards(self, user_id: int) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.id, b.title, b.created_at, b.updated_at, COUNT(m.id) AS message_count
                FROM chat_boards b
                LEFT JOIN chat_messages m ON m.board_id = b.id
                WHERE b.user_id = ?
                GROUP BY b.id
                ORDER BY b.updated_at DESC, b.id DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_board(self, user_id: int, title: str = "New chat") -> dict:
        now = int(time.time())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO chat_boards (user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, title, now, now),
            )
            board_id = cursor.lastrowid
        return {
            "id": board_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def get_board(self, user_id: int, board_id: int) -> dict:
        with self._connect() as connection:
            board = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_boards WHERE id = ? AND user_id = ?
                """,
                (board_id, user_id),
            ).fetchone()
            if board is None:
                self._not_found()
            messages = connection.execute(
                """
                SELECT id, role, content, citations, source, media_error, created_at
                FROM chat_messages WHERE board_id = ? ORDER BY created_at, id
                """,
                (board_id,),
            ).fetchall()
            media_rows = connection.execute(
                """
                SELECT cm.id, cm.message_id, cm.media_type AS type, cm.mime_type, cm.alt
                FROM chat_media cm
                JOIN chat_messages m ON m.id = cm.message_id
                WHERE m.board_id = ? ORDER BY cm.id
                """,
                (board_id,),
            ).fetchall()
            attachment_rows = connection.execute(
                """
                SELECT ca.id, ca.message_id, ca.filename, ca.mime_type, ca.size, ca.preview_type
                FROM chat_attachments ca
                JOIN chat_messages m ON m.id = ca.message_id
                WHERE m.board_id = ? ORDER BY ca.id
                """,
                (board_id,),
            ).fetchall()
        result = dict(board)
        media_by_message: dict[int, list[dict]] = {}
        for media in media_rows:
            item = dict(media)
            message_id = item.pop("message_id")
            media_by_message.setdefault(message_id, []).append(item)
        attachments_by_message: dict[int, list[dict]] = {}
        for attachment in attachment_rows:
            item = dict(attachment)
            message_id = item.pop("message_id")
            attachments_by_message.setdefault(message_id, []).append(item)
        result["messages"] = [
            {
                **dict(message),
                "citations": json.loads(message["citations"]),
                "media_error": (
                    json.loads(message["media_error"]) if message["media_error"] else None
                ),
                "media": media_by_message.get(message["id"], []),
                "attachments": attachments_by_message.get(message["id"], []),
            }
            for message in messages
        ]
        return result

    def rename_board(self, user_id: int, board_id: int, title: str) -> dict:
        now = int(time.time())
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_boards SET title = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (title, now, board_id, user_id),
            )
            if cursor.rowcount == 0:
                self._not_found()
        return self.get_board(user_id, board_id)

    def delete_board(self, user_id: int, board_id: int) -> None:
        with self._connect() as connection:
            media = connection.execute(
                """
                SELECT cm.filename FROM chat_media cm
                JOIN chat_messages m ON m.id = cm.message_id
                JOIN chat_boards b ON b.id = m.board_id
                WHERE b.id = ? AND b.user_id = ?
                """,
                (board_id, user_id),
            ).fetchall()
            attachments = connection.execute(
                """
                SELECT ca.stored_filename FROM chat_attachments ca
                JOIN chat_messages m ON m.id = ca.message_id
                JOIN chat_boards b ON b.id = m.board_id
                WHERE b.id = ? AND b.user_id = ?
                """,
                (board_id, user_id),
            ).fetchall()
            cursor = connection.execute(
                "DELETE FROM chat_boards WHERE id = ? AND user_id = ?",
                (board_id, user_id),
            )
            if cursor.rowcount == 0:
                self._not_found()
        for item in media:
            (self.media_path / Path(item["filename"]).name).unlink(missing_ok=True)
        for item in attachments:
            (self.attachment_path / Path(item["stored_filename"]).name).unlink(missing_ok=True)

    def search_messages(self, user_id: int, query: str, limit: int = 50) -> list[dict]:
        normalized = query.strip().lower()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.id AS board_id, b.title AS board_title,
                       m.id AS message_id, m.role, m.content, m.created_at
                FROM chat_messages m
                JOIN chat_boards b ON b.id = m.board_id
                WHERE b.user_id = ? AND instr(lower(m.content), ?) > 0
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT ?
                """,
                (user_id, normalized, limit),
            ).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            content = item.pop("content")
            match_at = content.lower().find(normalized)
            start = max(0, match_at - 60)
            end = min(len(content), match_at + len(normalized) + 90)
            snippet = content[start:end].replace("\n", " ").strip()
            item["snippet"] = ("…" if start else "") + snippet + ("…" if end < len(content) else "")
            results.append(item)
        return results

    def recent_messages(self, user_id: int, board_id: int, limit: int = 19) -> list[dict]:
        with self._connect() as connection:
            board = connection.execute(
                "SELECT id FROM chat_boards WHERE id = ? AND user_id = ?",
                (board_id, user_id),
            ).fetchone()
            if board is None:
                self._not_found()
            rows = connection.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content, created_at
                    FROM chat_messages
                    WHERE board_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                ORDER BY created_at, id
                """,
                (board_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message(
        self,
        user_id: int,
        board_id: int,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        source: str | None = None,
        media_error: dict | None = None,
    ) -> dict:
        now = int(time.time())
        message_source = (
            "user" if role == "user" else source or ("documentation" if citations else "general")
        )
        with self._connect() as connection:
            board = connection.execute(
                "SELECT title FROM chat_boards WHERE id = ? AND user_id = ?",
                (board_id, user_id),
            ).fetchone()
            if board is None:
                self._not_found()
            cursor = connection.execute(
                """
                INSERT INTO chat_messages
                    (board_id, role, content, citations, source, media_error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    board_id,
                    role,
                    content,
                    json.dumps(citations or []),
                    message_source,
                    json.dumps(media_error) if media_error else None,
                    now,
                ),
            )
            title = board["title"]
            if role == "user" and title == "New chat":
                title = content[:52].strip() + ("…" if len(content) > 52 else "")
            connection.execute(
                "UPDATE chat_boards SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, board_id),
            )
        return {
            "id": cursor.lastrowid,
            "role": role,
            "content": content,
            "citations": citations or [],
            "source": message_source,
            "created_at": now,
            "media": [],
            "attachments": [],
            "media_error": media_error,
        }

    def add_media(
        self,
        user_id: int,
        message_id: int,
        media_type: str,
        mime_type: str,
        data: bytes,
        alt: str,
    ) -> dict:
        extensions = {
            "image/png": ("image", ".png"),
            "image/jpeg": ("image", ".jpg"),
            "image/webp": ("image", ".webp"),
            "video/mp4": ("video", ".mp4"),
        }
        media_config = extensions.get(mime_type)
        if media_config is None or media_config[0] != media_type:
            raise ValueError("Unsupported generated media type.")
        extension = media_config[1]
        self.media_path.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{extension}"
        target = self.media_path / filename
        target.write_bytes(data)
        now = int(time.time())
        try:
            with self._connect() as connection:
                message = connection.execute(
                    """
                    SELECT m.id FROM chat_messages m
                    JOIN chat_boards b ON b.id = m.board_id
                    WHERE m.id = ? AND b.user_id = ? AND m.role = 'assistant'
                    """,
                    (message_id, user_id),
                ).fetchone()
                if message is None:
                    self._not_found()
                cursor = connection.execute(
                    """
                    INSERT INTO chat_media
                        (message_id, media_type, mime_type, filename, alt, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (message_id, media_type, mime_type, filename, alt, now),
                )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return {
            "id": cursor.lastrowid,
            "type": media_type,
            "mime_type": mime_type,
            "alt": alt,
        }

    def get_media(self, user_id: int, media_id: int) -> dict:
        with self._connect() as connection:
            media = connection.execute(
                """
                SELECT cm.mime_type, cm.filename
                FROM chat_media cm
                JOIN chat_messages m ON m.id = cm.message_id
                JOIN chat_boards b ON b.id = m.board_id
                WHERE cm.id = ? AND b.user_id = ?
                """,
                (media_id, user_id),
            ).fetchone()
        if media is None:
            self._not_found()
        path = self.media_path / Path(media["filename"]).name
        if not path.is_file():
            self._not_found()
        return {"path": path, "mime_type": media["mime_type"]}

    def add_attachment(
        self,
        user_id: int,
        message_id: int,
        filename: str,
        mime_type: str,
        data: bytes,
        preview_type: str,
        preview_text: str | None,
    ) -> dict:
        extension = Path(filename).suffix.lower()
        stored_filename = f"{uuid.uuid4().hex}{extension}"
        self.attachment_path.mkdir(parents=True, exist_ok=True)
        target = self.attachment_path / stored_filename
        target.write_bytes(data)
        now = int(time.time())
        try:
            with self._connect() as connection:
                message = connection.execute(
                    """
                    SELECT m.id FROM chat_messages m
                    JOIN chat_boards b ON b.id = m.board_id
                    WHERE m.id = ? AND b.user_id = ? AND m.role = 'user'
                    """,
                    (message_id, user_id),
                ).fetchone()
                if message is None:
                    self._not_found()
                cursor = connection.execute(
                    """
                    INSERT INTO chat_attachments
                        (message_id, filename, stored_filename, mime_type, size,
                         preview_type, preview_text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        filename,
                        stored_filename,
                        mime_type,
                        len(data),
                        preview_type,
                        preview_text,
                        now,
                    ),
                )
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return {
            "id": cursor.lastrowid,
            "filename": filename,
            "mime_type": mime_type,
            "size": len(data),
            "preview_type": preview_type,
        }

    def register_temporary_upload(
        self,
        user_id: int,
        filename: str,
        stored_filename: str,
        mime_type: str,
        size: int,
        preview_type: str,
    ) -> dict:
        upload_id = uuid.uuid4().hex
        now = int(time.time())
        expires_at = now + 3600
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO temporary_uploads
                    (id, user_id, filename, stored_filename, mime_type, size,
                     preview_type, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    user_id,
                    filename,
                    stored_filename,
                    mime_type,
                    size,
                    preview_type,
                    now,
                    expires_at,
                ),
            )
        return {
            "id": upload_id,
            "filename": filename,
            "mime_type": mime_type,
            "size": size,
            "preview_type": preview_type,
            "expires_at": expires_at,
        }

    def cleanup_temporary_uploads(self) -> None:
        now = int(time.time())
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT stored_filename FROM temporary_uploads WHERE expires_at <= ?", (now,)
            ).fetchall()
            connection.execute("DELETE FROM temporary_uploads WHERE expires_at <= ?", (now,))
        for row in rows:
            (self.temporary_upload_path / Path(row["stored_filename"]).name).unlink(missing_ok=True)

    def delete_temporary_upload(self, user_id: int, upload_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT stored_filename FROM temporary_uploads WHERE id = ? AND user_id = ?",
                (upload_id, user_id),
            ).fetchone()
            if row is None:
                self._not_found()
            connection.execute(
                "DELETE FROM temporary_uploads WHERE id = ? AND user_id = ?",
                (upload_id, user_id),
            )
        (self.temporary_upload_path / Path(row["stored_filename"]).name).unlink(missing_ok=True)

    def user_storage_bytes(self, user_id: int) -> int:
        with self._connect() as connection:
            persisted = connection.execute(
                """
                SELECT COALESCE(SUM(ca.size), 0)
                FROM chat_attachments ca
                JOIN chat_messages m ON m.id = ca.message_id
                JOIN chat_boards b ON b.id = m.board_id
                WHERE b.user_id = ?
                """,
                (user_id,),
            ).fetchone()[0]
            temporary = connection.execute(
                "SELECT COALESCE(SUM(size), 0) FROM temporary_uploads WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
        return int(persisted) + int(temporary)

    def get_temporary_uploads(self, user_id: int, upload_ids: list[str]) -> list[dict]:
        self.cleanup_temporary_uploads()
        if not upload_ids:
            return []
        if len(set(upload_ids)) != len(upload_ids):
            raise HTTPException(status_code=422, detail="Duplicate attachment IDs are not allowed.")
        placeholders = ",".join("?" for _ in upload_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, filename, stored_filename, mime_type, size, preview_type
                FROM temporary_uploads
                WHERE user_id = ? AND id IN ({placeholders})
                """,
                (user_id, *upload_ids),
            ).fetchall()
        by_id = {row["id"]: dict(row) for row in rows}
        if len(by_id) != len(upload_ids):
            raise HTTPException(status_code=404, detail="An attachment expired or was not found.")
        results = []
        for upload_id in upload_ids:
            item = by_id[upload_id]
            path = self.temporary_upload_path / Path(item.pop("stored_filename")).name
            if not path.is_file():
                raise HTTPException(status_code=404, detail="An attachment file was not found.")
            item["path"] = path
            results.append(item)
        return results

    def consume_temporary_upload(
        self,
        user_id: int,
        message_id: int,
        upload_id: str,
        preview_text: str | None,
    ) -> dict:
        upload = self.get_temporary_uploads(user_id, [upload_id])[0]
        extension = Path(upload["filename"]).suffix.lower()
        stored_filename = f"{uuid.uuid4().hex}{extension}"
        self.attachment_path.mkdir(parents=True, exist_ok=True)
        target = self.attachment_path / stored_filename
        upload["path"].replace(target)
        now = int(time.time())
        try:
            with self._connect() as connection:
                message = connection.execute(
                    """
                    SELECT m.id FROM chat_messages m
                    JOIN chat_boards b ON b.id = m.board_id
                    WHERE m.id = ? AND b.user_id = ? AND m.role = 'user'
                    """,
                    (message_id, user_id),
                ).fetchone()
                if message is None:
                    self._not_found()
                cursor = connection.execute(
                    """
                    INSERT INTO chat_attachments
                        (message_id, filename, stored_filename, mime_type, size,
                         preview_type, preview_text, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        upload["filename"],
                        stored_filename,
                        upload["mime_type"],
                        upload["size"],
                        upload["preview_type"],
                        preview_text,
                        now,
                    ),
                )
                connection.execute(
                    "DELETE FROM temporary_uploads WHERE id = ? AND user_id = ?",
                    (upload_id, user_id),
                )
        except Exception:
            target.replace(upload["path"])
            raise
        return {
            "id": cursor.lastrowid,
            "filename": upload["filename"],
            "mime_type": upload["mime_type"],
            "size": upload["size"],
            "preview_type": upload["preview_type"],
        }

    def get_attachment(self, user_id: int, attachment_id: int) -> dict:
        with self._connect() as connection:
            attachment = connection.execute(
                """
                SELECT ca.filename, ca.stored_filename, ca.mime_type
                FROM chat_attachments ca
                JOIN chat_messages m ON m.id = ca.message_id
                JOIN chat_boards b ON b.id = m.board_id
                WHERE ca.id = ? AND b.user_id = ?
                """,
                (attachment_id, user_id),
            ).fetchone()
        if attachment is None:
            self._not_found()
        path = self.attachment_path / Path(attachment["stored_filename"]).name
        if not path.is_file():
            self._not_found()
        return {
            "path": path,
            "filename": attachment["filename"],
            "mime_type": attachment["mime_type"],
        }

    def get_attachment_preview(self, user_id: int, attachment_id: int) -> dict:
        with self._connect() as connection:
            attachment = connection.execute(
                """
                SELECT ca.filename, ca.preview_text
                FROM chat_attachments ca
                JOIN chat_messages m ON m.id = ca.message_id
                JOIN chat_boards b ON b.id = m.board_id
                WHERE ca.id = ? AND b.user_id = ?
                """,
                (attachment_id, user_id),
            ).fetchone()
        if attachment is None:
            self._not_found()
        if attachment["preview_text"] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Text preview is unavailable."
            )
        return {"filename": attachment["filename"], "content": attachment["preview_text"]}

    @staticmethod
    def _not_found() -> None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat board not found.",
        )
