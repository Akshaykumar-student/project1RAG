import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge_base"
CONFIG_PATH = ROOT / "vector_store.json"


def existing_store_id() -> str | None:
    env_value = os.getenv("OPENAI_VECTOR_STORE_ID")
    if env_value:
        return env_value
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("vector_store_id")
        except json.JSONDecodeError:
            return None
    return None


def ingest(recreate: bool = False) -> str:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")

    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not files:
        raise RuntimeError(f"No Markdown files found in {KNOWLEDGE_DIR}")

    client = OpenAI()
    vector_store_id = None if recreate else existing_store_id()
    if vector_store_id:
        try:
            client.vector_stores.retrieve(vector_store_id)
            print(f"Reusing vector store {vector_store_id}")
        except Exception:
            vector_store_id = None

    if not vector_store_id:
        store = client.vector_stores.create(name="FlowDesk Help Center")
        vector_store_id = store.id
        print(f"Created vector store {vector_store_id}")

    existing_names: set[str] = set()
    if not recreate:
        existing = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100)
        for item in existing.data:
            try:
                existing_names.add(client.files.retrieve(item.id).filename)
            except Exception:
                continue

    pending_files = [path for path in files if path.name not in existing_names]
    if not pending_files:
        CONFIG_PATH.write_text(
            json.dumps({"vector_store_id": vector_store_id}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"All {len(files)} articles are already indexed; nothing to upload.")
        return vector_store_id

    streams = [path.open("rb") for path in pending_files]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=streams,
        )
    finally:
        for stream in streams:
            stream.close()

    if batch.status != "completed":
        raise RuntimeError(f"Indexing finished with status {batch.status}: {batch.file_counts}")

    CONFIG_PATH.write_text(
        json.dumps({"vector_store_id": vector_store_id}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Indexed {len(pending_files)} new articles; {len(existing_names)} already existed.")
    print(f"Saved vector store ID to {CONFIG_PATH.name}.")
    return vector_store_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload the FlowDesk help center to OpenAI.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Create a new vector store instead of reusing the configured store.",
    )
    args = parser.parse_args()
    try:
        ingest(recreate=args.recreate)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
