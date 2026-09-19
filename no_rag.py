import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


# ---------------------------------------------------------------------
# Terminal observability
# ---------------------------------------------------------------------
load_dotenv()

DEBUG = os.getenv("DEBUG", "1").lower() not in {"0", "false", "no", "off"}
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def paint(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def log(stage: str, message: str, *, color: str = "cyan", always: bool = False) -> None:
    if not DEBUG and not always:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    label = paint(f"[{timestamp}] [{stage.upper():>9}]", color)
    print(f"{label} {message}", flush=True)


def elapsed(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.2f}s"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
    )

CHAT_MODEL = "meta/muse-spark-1.3-contributor"
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
SUPPORTED_EXTENSIONS = {".txt", ".md"}

log("config", f"Chat model: {CHAT_MODEL}")
log("config", f"Documents directory: {DOCUMENTS_DIR}")


# ---------------------------------------------------------------------
# Load every supported document in full
# ---------------------------------------------------------------------
def load_full_documents() -> tuple[str, list[Path]]:
    """Read all supported files without splitting, embedding, or retrieval."""
    if not DOCUMENTS_DIR.exists():
        raise RuntimeError(f"Documents directory does not exist: {DOCUMENTS_DIR}")

    document_paths = sorted(
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not document_paths:
        raise RuntimeError(f"No supported documents found in {DOCUMENTS_DIR}")

    sections = []
    for index, file_path in enumerate(document_paths, start=1):
        text = file_path.read_text(encoding="utf-8")
        relative_source = file_path.relative_to(PROJECT_DIR)
        sections.append(
            f"[Document {index}]\n"
            f"Source: {relative_source}\n"
            f"--- BEGIN FILE ---\n"
            f"{text}\n"
            f"--- END FILE ---"
        )
        log("load", f"Loaded {relative_source} in full ({len(text):,} characters)")

    return "\n\n".join(sections), document_paths


load_started = time.perf_counter()
FULL_DOCUMENT_CONTENT, DOCUMENT_PATHS = load_full_documents()
log(
    "load",
    (
        f"Loaded {len(DOCUMENT_PATHS)} complete file(s), "
        f"{len(FULL_DOCUMENT_CONTENT):,} total prompt characters, "
        f"in {elapsed(load_started)}"
    ),
    color="green",
)


# ---------------------------------------------------------------------
# Muse Spark client
# ---------------------------------------------------------------------
llm = ChatOpenRouter(
    model=CHAT_MODEL,
    temperature=0.3,
    max_retries=2,
    api_key=OPENROUTER_API_KEY,
    app_title="Muse Spark Full Document Reader",
    app_url="http://localhost",
)

SYSTEM_PROMPT = """You are an assistant that answers questions from complete source files.
You will receive the full, unabridged contents of every loaded file.

Instructions:
1. Base your answer on the supplied files.
2. Do not invent facts that are absent from the files.
3. If the files do not contain enough information, say so clearly.
4. Give a clear, concise answer.
5. Mention source file names when that helps the user verify the answer.
"""


def ask_muse_spark(question: str) -> str:
    """Send the user's question and all full document contents directly to Muse Spark."""
    user_prompt = f"""Question:
{question}

Complete file contents:
{FULL_DOCUMENT_CONTENT}

Answer:
"""

    started = time.perf_counter()
    log(
        "generate",
        (
            f"Calling {CHAT_MODEL} with {len(DOCUMENT_PATHS)} complete file(s); "
            f"request text size={len(user_prompt):,} characters"
        ),
    )

    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    model_name = response_metadata.get("model_name") or response_metadata.get("model")
    finish_reason = response_metadata.get("finish_reason")

    log("generate", f"Model response received in {elapsed(started)}", color="green")
    if model_name:
        log("model", f"Resolved model: {model_name}")
    if finish_reason:
        log("model", f"Finish reason: {finish_reason}")
    if usage:
        log("model", f"Token usage: {json.dumps(usage, default=str)}")

    content = response.content
    if isinstance(content, str):
        return content
    return str(content)


# ---------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------
print()
log(
    "ready",
    "Muse Spark loaded the complete file set. Type 'exit' to quit.",
    color="green",
    always=True,
)
print(paint("Tip: set DEBUG=0 for quiet mode.", "dim"))

while True:
    question = input("\nEnter your question: ").strip()

    if question.lower() in {"exit", "quit", "stop"}:
        log("exit", "Exiting program. Goodbye!", always=True)
        break

    if not question:
        print("Please enter a question.")
        continue

    request_started = time.perf_counter()
    print()
    log("request", "Sending the complete file contents to Muse Spark", color="yellow", always=True)

    try:
        answer = ask_muse_spark(question)
        log(
            "request",
            f"Request completed in {elapsed(request_started)}",
            color="green",
            always=True,
        )
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 90)
    except Exception as exc:
        log(
            "error",
            f"Request failed after {elapsed(request_started)}: {type(exc).__name__}: {exc}",
            color="red",
            always=True,
        )
        if DEBUG:
            import traceback

            traceback.print_exc()
