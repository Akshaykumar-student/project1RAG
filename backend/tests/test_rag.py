import base64
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from backend.rag import (
    FALLBACK_ANSWER,
    MediaGenerationError,
    RAGConfigurationError,
    RAGService,
    classify_media_api_error,
    extract_citations,
    select_tools,
)

pytestmark = pytest.mark.asyncio


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


class FakeSequentialResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeStream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event


class FakeStreamingResponses:
    def __init__(self, events):
        self.events = events
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.events)


class FakeImages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def generate(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeVideoContent:
    async def aread(self):
        return b"mp4"


class FakeVideos:
    def __init__(self):
        self.create_kwargs = None
        self.downloaded_id = None

    async def create_and_poll(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id="video_test", status="completed")

    async def download_content(self, video_id):
        self.downloaded_id = video_id
        return FakeVideoContent()


def response_with_citations():
    return SimpleNamespace(
        output_text="Invitations expire after seven days.",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        annotations=[
                            SimpleNamespace(
                                type="file_citation", filename="02-invite-team-members.md"
                            ),
                            SimpleNamespace(
                                type="file_citation", filename="02-invite-team-members.md"
                            ),
                        ]
                    )
                ],
            )
        ],
    )


async def test_service_calls_file_search_and_extracts_unique_citations():
    fake = FakeClient(response_with_citations())
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)
    result = await service.ask("When does an invitation expire?")

    assert result.answer == "Invitations expire after seven days."
    assert result.citations[0].label == "Invite Team Members"
    assert len(result.citations) == 1
    assert result.source == "documentation"
    assert fake.responses.kwargs["model"] == "gpt-5-mini"
    assert fake.responses.kwargs["tools"][0]["max_num_results"] == 5
    assert len(fake.responses.kwargs["tools"]) == 1
    assert fake.responses.kwargs["tools"][0]["type"] == "file_search"
    assert fake.responses.kwargs["input"][-1] == {
        "role": "user",
        "content": "When does an invitation expire?",
    }
    assert fake.responses.kwargs["store"] is False
    assert fake.responses.kwargs["reasoning"] == {"effort": "low"}
    assert fake.responses.kwargs["max_output_tokens"] == 1200
    assert fake.responses.kwargs["tool_choice"] == "required"
    assert "generic SaaS advice" in fake.responses.kwargs["instructions"]
    assert "Do not append an offer to help" in fake.responses.kwargs["instructions"]


async def test_token_limited_empty_response_retries_with_more_output_space():
    incomplete = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_text="",
        output=[SimpleNamespace(type="file_search_call")],
        usage=SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=20),
            output_tokens=600,
        ),
    )
    completed = SimpleNamespace(
        status="completed",
        incomplete_details=None,
        output_text="FD-401 means the API key is missing or invalid.",
        output=[],
        usage=SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=100),
            output_tokens=150,
        ),
    )
    responses = FakeSequentialResponses([incomplete, completed])
    fake = SimpleNamespace(responses=responses)
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)

    result = await service.ask("How do I fix FD-401?")

    assert result.answer.startswith("FD-401")
    assert len(responses.calls) == 2
    assert responses.calls[1]["max_output_tokens"] == 2400
    assert service.last_usage == {
        "input_tokens": 200,
        "cached_input_tokens": 120,
        "output_tokens": 750,
    }
    assert service.last_file_search_calls == 1


async def test_service_requires_configuration():
    with pytest.raises(RAGConfigurationError):
        await RAGService(None, None).ask("question")


async def test_empty_model_output_uses_safe_fallback():
    response = SimpleNamespace(output_text="", output=[])
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: FakeClient(response))
    result = await service.ask("Unknown question")
    assert result.answer == FALLBACK_ANSWER
    assert result.citations == []
    assert result.source == "unavailable"


async def test_uncited_general_answer_is_labeled_general():
    response = SimpleNamespace(output_text="SQLite is a lightweight database.", output=[])
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: FakeClient(response))
    result = await service.ask("What is SQLite?")
    assert result.answer == "SQLite is a lightweight database."
    assert result.citations == []
    assert result.source == "general"


async def test_history_is_limited_to_10_messages_plus_current_question():
    fake = FakeClient(SimpleNamespace(output_text="Answer", output=[]))
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index)}
        for index in range(25)
    ]

    await service.ask("current", history=history)

    sent = fake.responses.kwargs["input"]
    assert len(sent) == 11
    assert sent[0]["content"] == "15"
    assert sent[-1]["content"] == "current"


async def test_general_questions_do_not_enable_search_tools():
    fake = FakeClient(SimpleNamespace(output_text="Four", output=[]))
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)

    await service.ask("What is two plus two?")

    assert fake.responses.kwargs["tools"] == []
    assert "include" not in fake.responses.kwargs
    assert "tool_choice" not in fake.responses.kwargs


@pytest.mark.parametrize(
    "question",
    [
        "Where can the owner rename the workspace?",
        "Which role can transfer workspace ownership?",
        "How many seats are included in Starter?",
        "When does a plan upgrade take effect?",
        "How many 2FA recovery codes are provided?",
        "Who can connect a support email inbox?",
        "Can a support agent reply to customers directly from Slack?",
        "What column is required for a customer CSV import?",
        "Where can an admin create an API key?",
        "Which plans include automation rules?",
        "My invitee waited over a week. Why can't they join?",
        "Will deleting a user lower next month's bill automatically?",
        "I paid for a monthly plan yesterday. Can I get my money back?",
        "I use Google to log in. Why am I not receiving a reset email?",
        "Why are replies coming from a generated address?",
        "Can I pay with cryptocurrency?",
    ],
)
async def test_implicit_support_questions_enable_file_search(question):
    assert select_tools(question) == ["file_search"]


async def test_support_follow_up_inherits_file_search_from_recent_history():
    history = [
        {"role": "user", "content": "Tell me about FlowDesk billing."},
        {"role": "assistant", "content": "What would you like to know?"},
    ]

    assert select_tools("What about annual plans?", history=history) == ["file_search"]


async def test_attachments_do_not_enable_unrelated_search_tools():
    assert select_tools("Summarize this FlowDesk invoice", has_attachments=True) == []


async def test_current_questions_enable_only_web_search():
    fake = FakeClient(SimpleNamespace(output_text="Current answer", output=[]))
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)

    await service.ask("What is the latest weather news?")

    assert fake.responses.kwargs["tools"] == [{"type": "web_search"}]


async def test_mixed_current_flowdesk_questions_enable_both_tools():
    fake = FakeClient(SimpleNamespace(output_text="Answer", output=[]))
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)

    await service.ask("Search the web for the latest FlowDesk billing information")

    assert [tool["type"] for tool in fake.responses.kwargs["tools"]] == [
        "file_search",
        "web_search",
    ]


async def test_stream_answer_emits_text_deltas_and_completion():
    completed = SimpleNamespace(output_text="Fast answer", output=[])
    responses = FakeStreamingResponses(
        [
            SimpleNamespace(type="response.output_text.delta", delta="Fast "),
            SimpleNamespace(type="response.output_text.delta", delta="answer"),
            SimpleNamespace(type="response.completed", response=completed),
        ]
    )
    fake = SimpleNamespace(responses=responses)
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)

    events = [event async for event in service.stream_answer("Explain SQLite")]

    assert [event["type"] for event in events] == ["delta", "delta", "complete"]
    assert events[-1]["response"].answer == "Fast answer"
    assert responses.kwargs["stream"] is True
    assert responses.kwargs["tools"] == []


async def test_attachment_parts_are_sent_only_with_current_message():
    fake = FakeClient(SimpleNamespace(output_text="Answer", output=[]))
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)
    attachment = {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "auto"}

    await service.ask(
        "What is shown?",
        history=[{"role": "user", "content": "Earlier"}],
        attachment_parts=[attachment],
    )

    assert fake.responses.kwargs["input"][0]["content"] == "Earlier"
    assert fake.responses.kwargs["input"][-1]["content"] == [
        {"type": "input_text", "text": "What is shown?"},
        attachment,
    ]


async def test_extracts_web_citations_and_labels_answer_as_web():
    response = SimpleNamespace(
        output_text="Current information.",
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                url="https://example.com/current",
                                title="Current source",
                            )
                        ]
                    )
                ],
            )
        ],
    )
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: FakeClient(response))

    result = await service.ask("What is current?")

    assert result.source == "web"
    assert result.citations[0].url == "https://example.com/current"
    assert result.citations[0].source_type == "web"


async def test_extract_citations_accepts_dictionary_responses():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"annotations": [{"type": "file_citation", "filename": "06-cancel-refunds.md"}]}
                ],
            }
        ]
    }
    assert extract_citations(response)[0].filename == "06-cancel-refunds.md"


async def test_generates_and_decodes_an_image():
    fake = SimpleNamespace(
        images=FakeImages(
            SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"png").decode())])
        )
    )
    service = RAGService("test-key", "vs_test", client_factory=lambda **_: fake)

    result = await service.generate_image("Create an image of a red kite")

    assert result.data == b"png"
    assert result.mime_type == "image/png"
    assert fake.images.kwargs["model"] == "gpt-image-2"
    assert fake.images.kwargs["size"] == "1024x1024"


async def test_video_generation_reports_retired_provider():
    service = RAGService("test-key", "vs_test")

    with pytest.raises(MediaGenerationError) as error:
        await service.generate_video("Create a video of a red kite")

    assert error.value.code == "video_unavailable"
    assert error.value.retryable is False


@pytest.mark.parametrize(
    ("status", "code", "expected", "retryable"),
    [
        (429, "insufficient_quota", "insufficient_quota", False),
        (403, "permission_denied", "model_access_denied", False),
        (400, "content_policy_violation", "safety_rejected", False),
        (429, "rate_limit_exceeded", "rate_limit", True),
        (500, "server_error", "provider_error", True),
    ],
)
async def test_classifies_media_provider_errors(status, code, expected, retryable):
    request = httpx.Request("POST", "https://api.openai.com/v1/images/generations")
    response = httpx.Response(status, request=request)
    provider_error = APIStatusError(
        "provider failure", response=response, body={"error": {"code": code}}
    )

    error = classify_media_api_error(provider_error, "image")

    assert error.code == expected
    assert error.retryable is retryable
