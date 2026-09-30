import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import ingest as ingest_module


class FakeFiles:
    def __init__(self, names=None, failures=None):
        self.names = names or {}
        self.failures = failures or set()

    def retrieve(self, file_id):
        if file_id in self.failures:
            raise RuntimeError("file lookup failed")
        return SimpleNamespace(filename=self.names[file_id])


class FakeFileBatches:
    def __init__(self, status="completed"):
        self.status = status
        self.uploaded_names: list[str] = []
        self.streams = []

    def upload_and_poll(self, *, vector_store_id, files):
        assert vector_store_id
        self.streams = list(files)
        self.uploaded_names = [Path(stream.name).name for stream in self.streams]
        return SimpleNamespace(status=self.status, file_counts={"completed": len(files)})


class FakeVectorStores:
    def __init__(self, *, retrieve_fails=False, listed_ids=(), batch_status="completed"):
        self.retrieve_fails = retrieve_fails
        self.listed_ids = listed_ids
        self.file_batches = FakeFileBatches(batch_status)
        self.files = SimpleNamespace(list=self.list_files)
        self.created = False

    def retrieve(self, store_id):
        if self.retrieve_fails:
            raise RuntimeError("missing store")
        return SimpleNamespace(id=store_id)

    def create(self, *, name):
        assert name == "FlowDesk Help Center"
        self.created = True
        return SimpleNamespace(id="vs_created")

    def list_files(self, *, vector_store_id, limit):
        assert vector_store_id
        assert limit == 100
        return SimpleNamespace(data=[SimpleNamespace(id=value) for value in self.listed_ids])


class FakeOpenAI:
    def __init__(self, vector_stores, files=None):
        self.vector_stores = vector_stores
        self.files = files or FakeFiles()


@pytest.fixture
def isolated_ingest(monkeypatch, tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    config = tmp_path / "vector_store.json"
    monkeypatch.setattr(ingest_module, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(ingest_module, "CONFIG_PATH", config)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_VECTOR_STORE_ID", raising=False)
    return knowledge, config


def test_existing_store_id_prefers_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_VECTOR_STORE_ID", "vs_environment")
    monkeypatch.setattr(ingest_module, "CONFIG_PATH", tmp_path / "missing.json")

    assert ingest_module.existing_store_id() == "vs_environment"


def test_existing_store_id_reads_valid_config_and_ignores_bad_json(monkeypatch, tmp_path):
    config = tmp_path / "vector_store.json"
    monkeypatch.delenv("OPENAI_VECTOR_STORE_ID", raising=False)
    monkeypatch.setattr(ingest_module, "CONFIG_PATH", config)
    config.write_text('{"vector_store_id":"vs_file"}', encoding="utf-8")
    assert ingest_module.existing_store_id() == "vs_file"

    config.write_text("not json", encoding="utf-8")
    assert ingest_module.existing_store_id() is None


def test_existing_store_id_returns_none_without_environment_or_config(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_VECTOR_STORE_ID", raising=False)
    monkeypatch.setattr(ingest_module, "CONFIG_PATH", tmp_path / "missing.json")

    assert ingest_module.existing_store_id() is None


def test_ingest_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ingest_module, "load_dotenv", lambda *_: None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is missing"):
        ingest_module.ingest()


def test_ingest_requires_markdown_articles(isolated_ingest):
    with pytest.raises(RuntimeError, match="No Markdown files"):
        ingest_module.ingest()


def test_ingest_reuses_store_and_skips_already_indexed_files(
    isolated_ingest, monkeypatch, capsys
):
    knowledge, config = isolated_ingest
    (knowledge / "one.md").write_text("one", encoding="utf-8")
    config.write_text(json.dumps({"vector_store_id": "vs_existing"}), encoding="utf-8")
    stores = FakeVectorStores(listed_ids=("file_1",))
    client = FakeOpenAI(stores, FakeFiles({"file_1": "one.md"}))
    monkeypatch.setattr(ingest_module, "OpenAI", lambda: client)

    result = ingest_module.ingest()

    assert result == "vs_existing"
    assert stores.created is False
    assert stores.file_batches.uploaded_names == []
    assert json.loads(config.read_text(encoding="utf-8"))["vector_store_id"] == result
    assert "nothing to upload" in capsys.readouterr().out


def test_ingest_creates_store_uploads_pending_files_and_closes_streams(
    isolated_ingest, monkeypatch
):
    knowledge, config = isolated_ingest
    (knowledge / "one.md").write_text("one", encoding="utf-8")
    (knowledge / "two.md").write_text("two", encoding="utf-8")
    stores = FakeVectorStores(retrieve_fails=True)
    client = FakeOpenAI(stores)
    monkeypatch.setattr(ingest_module, "OpenAI", lambda: client)
    monkeypatch.setenv("OPENAI_VECTOR_STORE_ID", "vs_deleted")

    result = ingest_module.ingest()

    assert result == "vs_created"
    assert stores.created is True
    assert stores.file_batches.uploaded_names == ["one.md", "two.md"]
    assert all(stream.closed for stream in stores.file_batches.streams)
    assert json.loads(config.read_text(encoding="utf-8")) == {"vector_store_id": "vs_created"}


def test_ingest_ignores_unreadable_existing_file_and_reports_failed_batch(
    isolated_ingest, monkeypatch
):
    knowledge, _ = isolated_ingest
    (knowledge / "one.md").write_text("one", encoding="utf-8")
    stores = FakeVectorStores(listed_ids=("broken",), batch_status="failed")
    client = FakeOpenAI(stores, FakeFiles(failures={"broken"}))
    monkeypatch.setattr(ingest_module, "OpenAI", lambda: client)
    monkeypatch.setenv("OPENAI_VECTOR_STORE_ID", "vs_existing")

    with pytest.raises(RuntimeError, match="Indexing finished with status failed"):
        ingest_module.ingest()

    assert all(stream.closed for stream in stores.file_batches.streams)


def test_recreate_always_creates_a_new_store(isolated_ingest, monkeypatch):
    knowledge, _ = isolated_ingest
    (knowledge / "one.md").write_text("one", encoding="utf-8")
    stores = FakeVectorStores()
    monkeypatch.setattr(ingest_module, "OpenAI", lambda: FakeOpenAI(stores))
    monkeypatch.setenv("OPENAI_VECTOR_STORE_ID", "vs_existing")

    assert ingest_module.ingest(recreate=True) == "vs_created"
    assert stores.created is True
