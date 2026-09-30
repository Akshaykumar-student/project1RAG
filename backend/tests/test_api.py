import base64
import json
from pathlib import Path

import pytest

from backend.auth import User
from backend.main import (
    app,
    get_chat_store,
    get_current_user,
    get_rag_service,
    is_image_request,
    is_video_request,
)
from backend.rag import GeneratedImage, GeneratedVideo, MediaGenerationError
from backend.schemas import ChatAttachment, ChatResponse, Citation

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def authenticated_user():
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, email="test@example.com", name="Test User"
    )
    yield
    app.dependency_overrides.clear()


class FakeRAGService:
    def __init__(self):
        self.history = None

    async def ask(self, question: str, history=None, attachment_parts=None) -> ChatResponse:
        self.history = history
        self.attachment_parts = attachment_parts
        return ChatResponse(
            answer=f"Answer for: {question}",
            citations=[Citation(filename="02-invite-team-members.md", label="Invite Team Members")],
            source="documentation",
        )

    async def stream_answer(self, question: str, history=None, attachment_parts=None):
        self.history = history
        self.attachment_parts = attachment_parts
        yield {"type": "delta", "text": "Fast "}
        yield {"type": "delta", "text": "answer"}
        yield {
            "type": "complete",
            "response": ChatResponse(answer="Fast answer", citations=[], source="general"),
        }

    async def generate_image(self, prompt: str) -> GeneratedImage:
        return GeneratedImage(b"image", "image/png", f"Generated for {prompt}")

    async def generate_video(self, prompt: str) -> GeneratedVideo:
        return GeneratedVideo(b"video", "video/mp4", f"Generated for {prompt}")


class FakeChatStore:
    def __init__(self):
        self.messages = []
        self.recent_limit = None

    def add_message(
        self, user_id, board_id, role, content, citations=None, source=None, media_error=None
    ):
        message = {
            "id": len(self.messages) + 1,
            "role": role,
            "content": content,
            "citations": citations or [],
            "source": source or "user",
            "media_error": media_error,
        }
        self.messages.append(message)
        return message

    def add_media(self, user_id, message_id, media_type, mime_type, data, alt):
        return {"id": 9, "type": media_type, "mime_type": mime_type, "alt": alt}

    def add_attachment(
        self,
        user_id,
        message_id,
        filename,
        mime_type,
        data,
        preview_type,
        preview_text,
    ):
        return {
            "id": 10,
            "filename": filename,
            "mime_type": mime_type,
            "size": len(data),
            "preview_type": preview_type,
        }

    def search_messages(self, user_id, query):
        return [
            {
                "board_id": 1,
                "board_title": "Billing help",
                "message_id": 2,
                "role": "assistant",
                "snippet": f"Matching {query}",
                "created_at": 1,
            }
        ]

    def recent_messages(self, user_id, board_id, limit=19):
        self.recent_limit = limit
        return [{"role": "user", "content": "Earlier question"}]


async def test_health_reports_setup_state(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "setup_required"}
    assert response.json()["model"] == "gpt-5-mini"
    assert isinstance(response.json()["attachment_support"], bool)
    assert isinstance(response.json()["missing_attachment_dependencies"], list)


async def test_sample_questions(client):
    response = await client.get("/api/sample-questions")
    assert response.status_code == 200
    assert len(response.json()["questions"]) == 3


async def test_binary_upload_returns_temporary_attachment_metadata(client, tmp_path: Path):
    class UploadStore(FakeChatStore):
        temporary_upload_path = tmp_path

        def cleanup_temporary_uploads(self):
            return None

        def register_temporary_upload(
            self, user_id, filename, stored_filename, mime_type, size, preview_type
        ):
            return {
                "id": "upload-id",
                "filename": filename,
                "mime_type": mime_type,
                "size": size,
                "preview_type": preview_type,
                "expires_at": 9999999999,
            }

    app.dependency_overrides[get_chat_store] = lambda: UploadStore()
    try:
        response = await client.post(
            "/api/uploads",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
    finally:
        app.dependency_overrides.pop(get_chat_store, None)

    assert response.status_code == 201
    assert response.json()["id"] == "upload-id"
    assert response.json()["size"] == 5


async def test_chat_returns_answer_and_citations(client):
    service = FakeRAGService()
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_chat_store] = lambda: FakeChatStore()
    try:
        response = await client.post(
            "/api/chat", json={"board_id": 1, "question": "How do I invite someone?"}
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_chat_store, None)
    assert response.status_code == 200
    assert response.json()["citations"][0]["filename"] == "02-invite-team-members.md"
    assert response.json()["source"] == "documentation"
    assert service.history == [{"role": "user", "content": "Earlier question"}]


async def test_stream_chat_emits_deltas_then_persists_completed_messages(client):
    service = FakeRAGService()
    store = FakeChatStore()
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_chat_store] = lambda: store
    try:
        response = await client.post(
            "/api/chat/stream", json={"board_id": 1, "question": "Explain SQLite"}
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_chat_store, None)

    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["delta", "delta", "complete"]
    assert events[-1]["answer"] == "Fast answer"
    assert [message["role"] for message in store.messages] == ["user", "assistant"]
    assert store.messages[-1]["content"] == "Fast answer"
    assert store.recent_limit == 10


async def test_chat_generates_inline_image_metadata(client):
    service = FakeRAGService()
    store = FakeChatStore()
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_chat_store] = lambda: store
    try:
        response = await client.post(
            "/api/chat", json={"board_id": 1, "question": "image of a fox"}
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_chat_store, None)
    assert response.status_code == 200
    assert response.json()["media"][0]["type"] == "image"
    assert response.json()["media"][0]["id"] == 9
    assert store.messages[-1]["role"] == "assistant"
    assert service.history is None
    assert is_image_request("Show me how image compression works") is False


async def test_chat_generates_video_without_external_link(client):
    store = FakeChatStore()
    service = FakeRAGService()
    app.dependency_overrides[get_rag_service] = lambda: service
    app.dependency_overrides[get_chat_store] = lambda: store
    try:
        response = await client.post(
            "/api/chat", json={"board_id": 1, "question": "video of a fox"}
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_chat_store, None)
    assert response.status_code == 200
    assert response.json()["media"][0]["type"] == "video"
    assert response.json()["citations"] == []
    assert "http" not in response.json()["answer"]
    assert service.history is None


async def test_chat_generates_image_and_video_together(client):
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    app.dependency_overrides[get_chat_store] = lambda: FakeChatStore()
    try:
        response = await client.post(
            "/api/chat", json={"board_id": 1, "question": "create an image and video of a fox"}
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_chat_store, None)
    assert response.status_code == 200
    assert {item["type"] for item in response.json()["media"]} == {"image", "video"}
    assert response.json()["citations"] == []


async def test_retryable_media_failure_is_persisted_without_links(client):
    class RateLimitedService(FakeRAGService):
        async def generate_image(self, prompt: str) -> GeneratedImage:
            raise MediaGenerationError(
                "rate_limit",
                "The image-generation rate limit was reached. Try again in one minute.",
                True,
            )

    store = FakeChatStore()
    app.dependency_overrides[get_rag_service] = lambda: RateLimitedService()
    app.dependency_overrides[get_chat_store] = lambda: store
    try:
        response = await client.post(
            "/api/chat", json={"board_id": 1, "question": "image of a fox"}
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
        app.dependency_overrides.pop(get_chat_store, None)
    assert response.status_code == 200
    assert response.json()["media"] == []
    assert response.json()["citations"] == []
    assert response.json()["media_error"] == {"code": "rate_limit", "retryable": True}
    assert "http" not in response.json()["answer"]
    assert store.messages[-1]["media_error"]["code"] == "rate_limit"


@pytest.mark.parametrize(
    ("prompt", "image", "video"),
    [
        ("image of a tiger", True, False),
        ("give me a picture", True, False),
        ("photo please", True, False),
        ("show me images of mountains", True, False),
        ("video of a waterfall", False, True),
        ("give me videos of waterfalls", False, True),
        ("make an animation", False, True),
        ("create an image and video", True, True),
        ("how does image compression work?", False, False),
        ("explain video compression", False, False),
    ],
)
async def test_media_intent_routing(prompt, image, video):
    assert is_image_request(prompt) is image
    assert is_video_request(prompt) is video


async def test_chat_rejects_blank_question(client):
    response = await client.post("/api/chat", json={"board_id": 1, "question": "  "})
    assert response.status_code == 422


async def test_chat_rejects_oversized_question(client):
    response = await client.post("/api/chat", json={"board_id": 1, "question": "x" * 1001})
    assert response.status_code == 422


async def test_prepares_text_and_image_attachments():
    from backend.main import prepare_attachments

    text = ChatAttachment(
        filename="notes.txt", mime_type="text/plain", encoding="utf8", size=5, content="hello"
    )
    png = b"\x89PNG\r\n\x1a\nimage"
    image = ChatAttachment(
        filename="photo.png",
        mime_type="image/png",
        encoding="base64",
        size=len(png),
        content=base64.b64encode(png).decode(),
    )

    parts = prepare_attachments([text, image])

    assert parts[0]["type"] == "input_text"
    assert parts[1]["type"] == "input_image"
    assert parts[1]["image_url"].startswith("data:image/png;base64,")


async def test_rejects_unsupported_attachment_before_saving_message(client):
    response = await client.post(
        "/api/chat",
        json={
            "board_id": 1,
            "question": "Read this",
            "attachments": [
                {
                    "filename": "program.exe",
                    "mime_type": "application/octet-stream",
                    "encoding": "base64",
                    "size": 3,
                    "content": "YWJj",
                }
            ],
        },
    )
    assert response.status_code == 422
    assert "unsupported" in response.json()["detail"].lower()


async def test_search_returns_authenticated_message_results(client):
    app.dependency_overrides[get_chat_store] = lambda: FakeChatStore()
    try:
        response = await client.get("/api/search", params={"q": "refund"})
    finally:
        app.dependency_overrides.pop(get_chat_store, None)
    assert response.status_code == 200
    assert response.json()[0]["board_title"] == "Billing help"
    assert response.json()[0]["message_id"] == 2


@pytest.mark.parametrize("query", ["", " ", "x", "x" * 101])
async def test_search_rejects_invalid_queries(client, query):
    response = await client.get("/api/search", params={"q": query})
    assert response.status_code == 422
