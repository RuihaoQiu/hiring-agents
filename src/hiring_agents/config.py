import os
from pathlib import Path

RANDOM_SEED: int = 42

CANDIDATE_COUNT: int = 500
MESSY_CASE_RATIO: float = 0.15

GENERATION_MODEL: str = "gpt-4o-mini"
EXTRACTION_MODEL: str = "gpt-4o-mini"
SUMMARY_MODEL: str = "gpt-4o-mini"
NORMALIZE_MODEL: str = "gpt-4o-mini"
RERANK_MODEL: str = "gpt-4o-mini"
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIM: int = 1536

GENERATION_TEMPERATURE: float = 0.9
EXTRACTION_TEMPERATURE: float = 0.0
SUMMARY_TEMPERATURE: float = 0.0
NORMALIZE_TEMPERATURE: float = 0.0
RERANK_TEMPERATURE: float = 0.0

SUMMARY_WORD_MIN: int = 200
SUMMARY_WORD_MAX: int = 300

RETRIEVAL_TOP_K: int = 10
RERANK_TOP_K: int = 10
SKIP_RERANK: bool = os.getenv("SKIP_RERANK", "").lower() in ("1", "true", "yes")
INGEST_CONCURRENCY: int = 10
RERANK_CONCURRENCY: int = 5
RERANK_MAX_RETRIES: int = 1

SCORE_MIN: int = 1
SCORE_MAX: int = 5

SENIORITY_VOCAB: list[str] = ["junior", "mid", "senior", "staff", "principal"]
SENIORITY_INFER_MODEL: str = EXTRACTION_MODEL

LLM_MAX_ATTEMPTS: int = 3
LLM_RETRY_WAIT_MIN_SECONDS: int = 2
LLM_RETRY_WAIT_MAX_SECONDS: int = 10
EMBED_BATCH_SIZE: int = 100
GENERATION_LOG_EVERY: int = 25

AGENT_MODEL: str = "gpt-4o"
AGENT_TEMPERATURE: float = 0.3

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
REPORTS_DIR: Path = ROOT / "reports"

CANDIDATES_PATH: Path = DATA_DIR / "candidates.json"
GROUND_TRUTH_PATH: Path = DATA_DIR / "ground_truth.json"
INGESTED_PATH: Path = DATA_DIR / "ingested.json"
EMBEDDINGS_PATH: Path = DATA_DIR / "embeddings.npy"
QUERIES_PATH: Path = DATA_DIR / "queries.json"
LABELS_PATH: Path = DATA_DIR / "labels.json"
