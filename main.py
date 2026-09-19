import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, NotRequired
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------
# Terminal observability
# ---------------------------------------------------------------------
DEBUG = os.getenv("RAG_DEBUG", "1").lower() not in {"0", "false", "no", "off"}
SHOW_PREVIEWS = os.getenv("RAG_SHOW_PREVIEWS", "1").lower() not in {
    "0", "false", "no", "off"
}
PREVIEW_CHARS = int(os.getenv("RAG_PREVIEW_CHARS", "160"))
EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16"))
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None

COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
}


def paint(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def log(stage: str, message: str, *, color: str = "cyan", always: bool = False) -> None:
    """Print a timestamped pipeline event without exposing credentials."""
    if not DEBUG and not always:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    label = paint(f"[{timestamp}] [{stage.upper():>9}]", color)
    print(f"{label} {message}", flush=True)


def elapsed(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.2f}s"


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def source_name(document: Document) -> str:
    return str(document.metadata.get("source", "unknown"))


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to your environment or .env file."
    )
if not TYPESAFE_API_KEY:
    raise RuntimeError(
        "TYPESAFE_API_KEY is not set. Add it to your environment or .env file."
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TYPESAFE_SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"
CHAT_MODEL = "meta/muse-spark-1.3-contributor"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# Retrieve a broad candidate set, then retain the best chunks after Jev scoring.
RETRIEVAL_CANDIDATES = 15
RERANKED_DOCUMENTS = 5
TYPESAFE_MAX_RETRIES = 3
TYPESAFE_TIMEOUT_SECONDS = 30

log("config", f"Debug logging: {'on' if DEBUG else 'off'}; previews: {'on' if SHOW_PREVIEWS else 'off'}")
log("config", f"Embedding model: {EMBEDDING_MODEL}")
log("config", f"Chat model: {CHAT_MODEL}; reranker: {TYPESAFE_MODEL}")
log("config", f"Retrieval: top {RETRIEVAL_CANDIDATES} candidates -> top {RERANKED_DOCUMENTS} after reranking")


# ---------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------
class State(TypedDict):
    question: str
    candidates: NotRequired[List[Document]]
    context: NotRequired[List[Document]]
    answer: NotRequired[str]


# ---------------------------------------------------------------------
# Load the knowledge base
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
VECTOR_CACHE_DIR = PROJECT_DIR / ".vector_cache"
INDEX_METADATA_PATH = VECTOR_CACHE_DIR / "index_metadata.json"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INDEX_SCHEMA_VERSION = "1"


def calculate_index_fingerprint(file_paths: list[Path]) -> str:
    """Hash all files and settings that affect document embeddings."""
    digest = hashlib.sha256()
    digest.update(INDEX_SCHEMA_VERSION.encode("utf-8"))
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    digest.update(str(CHUNK_SIZE).encode("utf-8"))
    digest.update(str(CHUNK_OVERLAP).encode("utf-8"))
    for file_path in sorted(file_paths):
        digest.update(str(file_path.relative_to(PROJECT_DIR)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def load_cached_fingerprint() -> str | None:
    if not INDEX_METADATA_PATH.exists():
        return None
    try:
        return json.loads(
            INDEX_METADATA_PATH.read_text(encoding="utf-8")
        ).get("fingerprint")
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_fingerprint(fingerprint: str) -> None:
    """Save metadata only after the vector index is built successfully."""
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_METADATA_PATH.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "embedding_model": EMBEDDING_MODEL,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "schema_version": INDEX_SCHEMA_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


local_docs = []
document_paths = []

load_started = time.perf_counter()
log("load", f"Scanning {DOCUMENTS_DIR}")
for file_path in DOCUMENTS_DIR.rglob("*"):
    if file_path.suffix.lower() not in {".txt", ".md"}:
        continue
    document_paths.append(file_path)
    text = file_path.read_text(encoding="utf-8")
    relative_source = str(file_path.relative_to(PROJECT_DIR))
    local_docs.append(
        Document(
            page_content=text,
            metadata={
                "source": relative_source,
                "filename": file_path.name,
                "file_type": file_path.suffix.lower(),
            },
        )
    )
    log("load", f"Loaded {relative_source} ({len(text):,} characters)")

if not local_docs:
    raise RuntimeError(f"No supported documents found in {DOCUMENTS_DIR}")
log("load", f"Loaded {len(local_docs)} document(s) in {elapsed(load_started)}", color="green")


# ---------------------------------------------------------------------
# Split documents
# ---------------------------------------------------------------------
split_started = time.perf_counter()
log("split", f"Splitting documents (chunk size {CHUNK_SIZE:,}; overlap {CHUNK_OVERLAP})")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
all_splits = text_splitter.split_documents(local_docs)
log("split", f"Created {len(all_splits)} chunks in {elapsed(split_started)}", color="green")


# ---------------------------------------------------------------------
# OpenRouter embeddings and persistent Chroma index
# ---------------------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
)

current_fingerprint = calculate_index_fingerprint(document_paths)
cached_fingerprint = load_cached_fingerprint()
cache_is_valid = VECTOR_CACHE_DIR.exists() and cached_fingerprint == current_fingerprint

if cache_is_valid:
    index_started = time.perf_counter()
    log("embed", f"Loading cached vector index from {VECTOR_CACHE_DIR}")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_CACHE_DIR),
    )
    log("embed", f"Cached vector index ready in {elapsed(index_started)}", color="green")
else:
    index_started = time.perf_counter()
    if VECTOR_CACHE_DIR.exists():
        log(
            "embed",
            "Documents or embedding settings changed; rebuilding the vector index",
            color="yellow",
        )
        shutil.rmtree(VECTOR_CACHE_DIR)

    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_CACHE_DIR),
    )
    total_batches = max(
        1,
        (len(all_splits) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE,
    )
    log("embed", f"Indexing {len(all_splits)} chunks in {total_batches} batch(es)")
    for batch_number, start in enumerate(
        range(0, len(all_splits), EMBEDDING_BATCH_SIZE),
        start=1,
    ):
        batch = all_splits[start : start + EMBEDDING_BATCH_SIZE]
        batch_started = time.perf_counter()
        end = start + len(batch)
        log("embed", f"Batch {batch_number}/{total_batches}: chunks {start + 1}-{end}")
        vector_store.add_documents(batch)
        log(
            "embed",
            f"Batch {batch_number}/{total_batches} complete in {elapsed(batch_started)}",
            color="green",
        )

    save_cached_fingerprint(current_fingerprint)
    log(
        "embed",
        f"Vector index built and cached in {elapsed(index_started)}",
        color="green",
    )

# ---------------------------------------------------------------------
# OpenRouter chat model
# ---------------------------------------------------------------------
llm = ChatOpenRouter(
    model=CHAT_MODEL,
    temperature=0.3,
    max_retries=2,
    api_key=OPENROUTER_API_KEY,
    app_title="LangGraph RAG Pipeline",
    app_url="http://localhost",
)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------
CUSTOM_PROMPT = """
You are an advanced retrieval-augmented assistant.
Answer the user's question using only the supplied context.

Instructions:
1. Do not invent facts that are absent from the context.
2. If the context is insufficient, clearly say that the knowledge base
   does not contain enough information.
3. Give a clear and concise answer.
4. When appropriate, explain which parts of the context support the answer.

Question:
{question}

Context:
{context}

Answer:
"""


# ---------------------------------------------------------------------
# TypeSafe Jev re-ranking
# ---------------------------------------------------------------------
def score_candidate_with_jev(question: str, candidate: Document) -> tuple[float, float]:
    """Score one candidate chunk for relevance using TypeSafe Jev."""
    payload = {
        "model": TYPESAFE_MODEL,
        "state": {
            "user_question": question,
            "candidate_passage": candidate.page_content,
            "candidate_source": source_name(candidate),
        },
        "questions": {
            "relevance": {
                "type": "score",
                "instructions": (
                    "Rate how useful this candidate passage is for answering the "
                    "user question. Judge direct topical relevance, whether the "
                    "passage contains answer-bearing facts, and whether it helps "
                    "answer the specific question rather than merely sharing words."
                ),
                "criteria": [
                    "Completely irrelevant or unrelated",
                    "Slightly related but not useful for answering the question",
                    "Moderately relevant and may provide supporting context",
                    "Highly relevant and contains useful answer-bearing information",
                    "Directly answers the question with specific supporting information",
                ],
            }
        },
    }
    encoded_payload = json.dumps(payload).encode("utf-8")

    for attempt in range(1, TYPESAFE_MAX_RETRIES + 1):
        # Build a fresh Request for every retry.
        request = Request(
            TYPESAFE_SYSTEMONE_URL,
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {TYPESAFE_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request_started = time.perf_counter()
        try:
            log("jev", f"Request attempt {attempt}/{TYPESAFE_MAX_RETRIES} for {source_name(candidate)}")
            with urlopen(request, timeout=TYPESAFE_TIMEOUT_SECONDS) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            relevance = response_data["answers"]["relevance"]
            score = float(relevance["score"])
            confidence = float(relevance.get("confidence", 0.0))
            log("jev", f"Received score={score:.3f}, confidence={confidence:.3f} in {elapsed(request_started)}", color="green")
            return score, confidence
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            log("jev", f"HTTP {exc.code} after {elapsed(request_started)}: {preview(error_body, 240)}", color="red")
            if exc.code not in {429, 529} or attempt == TYPESAFE_MAX_RETRIES:
                raise RuntimeError(
                    f"TypeSafe Jev request failed with HTTP {exc.code}: {error_body}"
                ) from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2 ** (attempt - 1)
            log("retry", f"Rate limited; waiting {delay:.1f}s before retry", color="yellow")
            time.sleep(delay)
        except (URLError, TimeoutError) as exc:
            log("jev", f"Network failure after {elapsed(request_started)}: {exc}", color="red")
            if attempt == TYPESAFE_MAX_RETRIES:
                raise RuntimeError(f"TypeSafe Jev request failed: {exc}") from exc
            delay = 2 ** (attempt - 1)
            log("retry", f"Waiting {delay:.1f}s before retry", color="yellow")
            time.sleep(delay)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("TypeSafe Jev returned an unexpected response shape.") from exc

    raise RuntimeError("TypeSafe Jev scoring failed after all retries.")


# ---------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------
def classify(state: State) -> dict:
    """Placeholder classification node."""
    log("classify", f"Received question: {state['question']!r}")
    return {"question": state["question"]}


def retrieve(state: State) -> dict:
    started = time.perf_counter()
    log("retrieve", f"Searching vector index for top {RETRIEVAL_CANDIDATES} candidates")
    candidates = vector_store.similarity_search(
        state["question"],
        k=RETRIEVAL_CANDIDATES,
    )
    log("retrieve", f"Found {len(candidates)} candidate(s) in {elapsed(started)}", color="green")
    for rank, candidate in enumerate(candidates, start=1):
        details = f"#{rank:02d} {source_name(candidate)} ({len(candidate.page_content):,} chars)"
        if SHOW_PREVIEWS:
            details += f" | {preview(candidate.page_content)}"
        log("candidate", details, color="magenta")
    return {"candidates": candidates}


def rerank(state: State) -> dict:
    """Present every retrieved candidate to Jev and rank by relevance score."""
    candidates = state.get("candidates", [])
    scored_candidates = []
    started = time.perf_counter()
    log("rerank", f"Scoring {len(candidates)} candidate(s) with {TYPESAFE_MODEL}")

    for retrieval_rank, candidate in enumerate(candidates, start=1):
        candidate_started = time.perf_counter()
        log("rerank", f"Scoring {retrieval_rank}/{len(candidates)}: {source_name(candidate)}")
        score, confidence = score_candidate_with_jev(state["question"], candidate)
        reranked_candidate = Document(
            page_content=candidate.page_content,
            metadata={
                **candidate.metadata,
                "retrieval_rank": retrieval_rank,
                "jev_relevance_score": score,
                "jev_confidence": confidence,
            },
        )
        scored_candidates.append(reranked_candidate)
        log("rerank", f"Finished {retrieval_rank}/{len(candidates)} in {elapsed(candidate_started)}", color="green")

    scored_candidates.sort(
        key=lambda document: (
            document.metadata["jev_relevance_score"],
            document.metadata["jev_confidence"],
            -document.metadata["retrieval_rank"],
        ),
        reverse=True,
    )

    log("ranking", "Final Jev ranking:", color="yellow")
    for rank, document in enumerate(scored_candidates, start=1):
        selected = "SELECTED" if rank <= RERANKED_DOCUMENTS else "dropped"
        log(
            "ranking",
            f"#{rank:02d} score={document.metadata['jev_relevance_score']:.3f} "
            f"confidence={document.metadata['jev_confidence']:.3f} "
            f"retrieval=#{document.metadata['retrieval_rank']:02d} "
            f"[{selected}] {source_name(document)}",
            color="green" if rank <= RERANKED_DOCUMENTS else "dim",
        )

    context = scored_candidates[:RERANKED_DOCUMENTS]
    log("rerank", f"Selected {len(context)} context chunk(s) in {elapsed(started)}", color="green")
    return {"context": context}


def generate(state: State) -> dict:
    context = state.get("context", [])
    docs_content = "\n\n".join(
        (
            f"[Document {index}]\n"
            f"Source: {source_name(document)}\n"
            f"Relevance score: {document.metadata.get('jev_relevance_score', 'not scored')}\n"
            f"{document.page_content}"
        )
        for index, document in enumerate(context, start=1)
    )
    if not docs_content:
        docs_content = "No relevant documents were retrieved."

    prompt = CUSTOM_PROMPT.format(question=state["question"], context=docs_content)
    started = time.perf_counter()
    log("generate", f"Calling {CHAT_MODEL} with {len(context)} chunks; prompt size={len(prompt):,} chars")
    response = llm.invoke([{"role": "user", "content": prompt}])

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
    elif response_metadata:
        log("model", f"Response metadata: {json.dumps(response_metadata, default=str)}")
    else:
        log("model", "Provider returned no response metadata", color="yellow")

    content = response.content
    if not isinstance(content, str):
        content = str(content)
    log("generate", f"Answer size: {len(content):,} characters")
    return {"answer": content}


def refine(state: State) -> dict:
    """Placeholder refinement node."""
    log("refine", "Passing generated answer through placeholder refinement node")
    answer = state.get("answer", "No answer was generated.")
    return {"answer": answer}


# ---------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("classify", classify)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("rerank", rerank)
graph_builder.add_node("generate", generate)
graph_builder.add_node("refine", refine)
graph_builder.add_edge(START, "classify")
graph_builder.add_edge("classify", "retrieve")
graph_builder.add_edge("retrieve", "rerank")
graph_builder.add_edge("rerank", "generate")
graph_builder.add_edge("generate", "refine")
graph_builder.add_edge("refine", END)
graph = graph_builder.compile()
log("graph", "Compiled: classify -> retrieve -> rerank -> generate -> refine", color="green")


# ---------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------
print()
log("ready", "RAG system is ready. Type 'exit' to quit.", color="green", always=True)
print(paint("Tip: set RAG_DEBUG=0 for quiet mode or RAG_SHOW_PREVIEWS=0 to hide chunk previews.", "dim"))

while True:
    question = input("\nEnter your question: ").strip()
    if question.lower() in {"exit", "quit", "stop"}:
        log("exit", "Exiting program. Goodbye!", always=True)
        break
    if not question:
        print("Please enter a question.")
        continue

    pipeline_started = time.perf_counter()
    print()
    log("pipeline", "Starting request", color="yellow", always=True)
    try:
        response = graph.invoke({"question": question})
        answer = response.get("answer", "No answer was generated.")
        log("pipeline", f"Request completed in {elapsed(pipeline_started)}", color="green", always=True)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 90)
    except Exception as exc:
        log("error", f"Pipeline failed after {elapsed(pipeline_started)}: {type(exc).__name__}: {exc}", color="red", always=True)
        if DEBUG:
            import traceback
            traceback.print_exc()
