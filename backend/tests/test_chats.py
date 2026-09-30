import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.auth import AuthService
from backend.chats import ChatStore


@pytest.fixture
def store(tmp_path: Path):
    database = str(tmp_path / "chats.db")
    auth = AuthService(database, "test-secret")
    first = auth.register("first@example.com", "First", "password-123")
    second = auth.register("second@example.com", "Second", "password-123")
    return (
        ChatStore(
            database,
            str(tmp_path / "media"),
            str(tmp_path / "attachments"),
            str(tmp_path / "temporary"),
        ),
        first,
        second,
    )


def test_board_contains_multiple_persisted_messages(store):
    chats, user, _ = store
    board = chats.create_board(user.id)
    chats.add_message(user.id, board["id"], "user", "How do I invite someone?")
    chats.add_message(
        user.id,
        board["id"],
        "assistant",
        "Use Team settings.",
        [{"filename": "02-invite-team-members.md", "label": "Invite Team Members"}],
    )

    detail = chats.get_board(user.id, board["id"])
    assert len(detail["messages"]) == 2
    assert detail["messages"][1]["citations"][0]["filename"].startswith("02-")
    assert detail["messages"][0]["source"] == "user"
    assert detail["messages"][1]["source"] == "documentation"
    assert detail["title"].startswith("How do I invite")


def test_boards_are_private_to_their_owner(store):
    chats, first, second = store
    board = chats.create_board(first.id, "Private board")
    with pytest.raises(HTTPException) as error:
        chats.get_board(second.id, board["id"])
    assert error.value.status_code == 404


def test_rename_and_delete_board(store):
    chats, user, _ = store
    board = chats.create_board(user.id)
    renamed = chats.rename_board(user.id, board["id"], "Billing questions")
    assert renamed["title"] == "Billing questions"
    chats.delete_board(user.id, board["id"])
    assert chats.list_boards(user.id) == []


def test_generated_media_is_private_persisted_and_deleted(store):
    chats, first, second = store
    board = chats.create_board(first.id)
    message = chats.add_message(first.id, board["id"], "assistant", "Generated image")
    media = chats.add_media(
        first.id, message["id"], "image", "image/png", b"fake-png", "A test image"
    )

    detail = chats.get_board(first.id, board["id"])
    assert detail["messages"][0]["media"] == [
        {"id": media["id"], "type": "image", "mime_type": "image/png", "alt": "A test image"}
    ]
    stored = chats.get_media(first.id, media["id"])
    assert stored["path"].read_bytes() == b"fake-png"
    video = chats.add_media(
        first.id, message["id"], "video", "video/mp4", b"fake-mp4", "A test video"
    )
    stored_video = chats.get_media(first.id, video["id"])
    assert stored_video["path"].read_bytes() == b"fake-mp4"
    with pytest.raises(HTTPException):
        chats.get_media(second.id, media["id"])

    chats.delete_board(first.id, board["id"])
    assert not stored["path"].exists()
    assert not stored_video["path"].exists()


def test_attachments_are_private_persisted_and_deleted(store):
    chats, first, second = store
    board = chats.create_board(first.id)
    message = chats.add_message(first.id, board["id"], "user", "Read my resume")
    attachment = chats.add_attachment(
        first.id,
        message["id"],
        "../My Resume.pdf",
        "application/pdf",
        b"fake-pdf",
        "pdf",
        None,
    )

    detail = chats.get_board(first.id, board["id"])
    assert detail["messages"][0]["content"] == "Read my resume"
    assert detail["messages"][0]["attachments"] == [
        {
            "id": attachment["id"],
            "filename": "../My Resume.pdf",
            "mime_type": "application/pdf",
            "size": 8,
            "preview_type": "pdf",
        }
    ]
    stored = chats.get_attachment(first.id, attachment["id"])
    assert stored["path"].parent.name == "attachments"
    assert stored["path"].read_bytes() == b"fake-pdf"
    with pytest.raises(HTTPException):
        chats.get_attachment(second.id, attachment["id"])

    chats.delete_board(first.id, board["id"])
    assert not stored["path"].exists()


def test_text_attachment_preview_is_private(store):
    chats, first, second = store
    board = chats.create_board(first.id)
    message = chats.add_message(first.id, board["id"], "user", "Summarize this")
    attachment = chats.add_attachment(
        first.id,
        message["id"],
        "notes.txt",
        "text/plain",
        b"Private notes",
        "text",
        "Private notes",
    )

    preview = chats.get_attachment_preview(first.id, attachment["id"])
    assert preview == {"filename": "notes.txt", "content": "Private notes"}
    with pytest.raises(HTTPException):
        chats.get_attachment_preview(second.id, attachment["id"])


def test_temporary_upload_is_private_and_consumed_without_copying(store):
    chats, first, second = store
    chats.temporary_upload_path.mkdir(parents=True)
    temporary_file = chats.temporary_upload_path / "temporary.pdf"
    temporary_file.write_bytes(b"%PDF-test")
    upload = chats.register_temporary_upload(
        first.id,
        "resume.pdf",
        temporary_file.name,
        "application/pdf",
        temporary_file.stat().st_size,
        "pdf",
    )
    with pytest.raises(HTTPException):
        chats.get_temporary_uploads(second.id, [upload["id"]])

    board = chats.create_board(first.id)
    message = chats.add_message(first.id, board["id"], "user", "Read this")
    attachment = chats.consume_temporary_upload(first.id, message["id"], upload["id"], None)

    assert attachment["filename"] == "resume.pdf"
    assert not temporary_file.exists()
    assert chats.get_attachment(first.id, attachment["id"])["path"].read_bytes() == b"%PDF-test"
    with pytest.raises(HTTPException):
        chats.get_temporary_uploads(first.id, [upload["id"]])


def test_media_error_persists_with_assistant_message(store):
    chats, user, _ = store
    board = chats.create_board(user.id)
    chats.add_message(
        user.id,
        board["id"],
        "assistant",
        "Rate limit reached.",
        source="unavailable",
        media_error={"code": "rate_limit", "retryable": True},
    )

    message = chats.get_board(user.id, board["id"])["messages"][0]

    assert message["media_error"] == {"code": "rate_limit", "retryable": True}
    assert message["media"] == []


def test_searches_questions_and_answers_without_leaking_other_users(store):
    chats, first, second = store
    first_board = chats.create_board(first.id, "Billing help")
    second_board = chats.create_board(second.id, "Private billing")
    chats.add_message(first.id, first_board["id"], "user", "Where is my REFUND?")
    chats.add_message(first.id, first_board["id"], "assistant", "Your refund takes five days.")
    chats.add_message(second.id, second_board["id"], "assistant", "Secret refund details")

    results = chats.search_messages(first.id, "refund")

    assert len(results) == 2
    assert {result["role"] for result in results} == {"user", "assistant"}
    assert all(result["board_id"] == first_board["id"] for result in results)
    assert all("snippet" in result for result in results)


def test_search_treats_sql_wildcards_as_literal_text(store):
    chats, user, _ = store
    board = chats.create_board(user.id)
    chats.add_message(user.id, board["id"], "user", "Usage reached 50% today")
    chats.add_message(user.id, board["id"], "assistant", "Nothing special")

    results = chats.search_messages(user.id, "50%")

    assert len(results) == 1
    assert "50%" in results[0]["snippet"]


def test_recent_messages_are_board_scoped_ordered_and_limited(store):
    chats, user, _ = store
    first = chats.create_board(user.id, "First")
    second = chats.create_board(user.id, "Second")
    for index in range(25):
        chats.add_message(user.id, first["id"], "user", f"Message {index}")
    chats.add_message(user.id, second["id"], "user", "Other board")

    history = chats.recent_messages(user.id, first["id"], limit=19)

    assert len(history) == 19
    assert history[0]["content"] == "Message 6"
    assert history[-1]["content"] == "Message 24"
    assert all(message["content"] != "Other board" for message in history)


def test_existing_database_messages_receive_sources(tmp_path: Path):
    database = tmp_path / "legacy.db"
    fallback = (
        "I don't have enough information in the FlowDesk documentation to answer that question."
    )
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.executescript(
                """
                CREATE TABLE chat_boards (
                    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, title TEXT NOT NULL,
                    created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                );
                CREATE TABLE chat_messages (
                    id INTEGER PRIMARY KEY, board_id INTEGER NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, citations TEXT NOT NULL DEFAULT '[]',
                    created_at INTEGER NOT NULL
                );
                INSERT INTO chat_boards VALUES (1, 7, 'Legacy', 1, 1);
                """
            )
            connection.execute(
                "INSERT INTO chat_messages VALUES (1, 1, 'assistant', 'Grounded', ?, 1)",
                (json.dumps([{"filename": "help.md", "label": "Help"}]),),
            )
            connection.execute(
                "INSERT INTO chat_messages VALUES (2, 1, 'assistant', ?, '[]', 2)",
                (fallback,),
            )

    chats = ChatStore(str(database))
    messages = chats.get_board(7, 1)["messages"]

    assert messages[0]["source"] == "documentation"
    assert messages[1]["source"] == "unavailable"
