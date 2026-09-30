import os

import pytest

from backend.config import Settings
from backend.rag import RAGService


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_VECTOR_STORE_ID")),
    reason="Live OpenAI credentials are not configured",
)
async def test_live_openai_file_search():
    settings = Settings()
    result = await RAGService(
        settings.openai_api_key,
        settings.vector_store_id,
        settings.openai_model,
    ).ask("How long does a password reset link last?")
    assert result.answer
    assert result.citations
