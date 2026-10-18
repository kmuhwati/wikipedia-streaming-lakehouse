"""Paths and settings for the streaming lakehouse, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LANDING = DATA / "landing"              # raw JSONL from the live stream (at-least-once)
LAKE = DATA / "lake"                    # Iceberg warehouse + SQLite catalog
INDEX = DATA / "index"                  # LanceDB chunk index
CACHE = DATA / "cache"                  # fetched revision text, so re-runs make no API calls
LIVE = DATA / "live"                    # separate lake + index for the live freshness run
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# ---- lakehouse ----
NAMESPACE = "wiki"
FILES_PER_COMMIT = 4                    # landing files per bronze micro-batch
OFFSET_KEY = "wikistream.landing.last-file"
BRONZE_BATCH_KEY = "wikistream.bronze.last-batch"
DEDUP_LOOKBACK_HOURS = 2                # SSE replays after a reconnect are minutes old, never hours

# ---- the article set kept fresh in the RAG index ----
RAG_WIKI = "enwiki"
RAG_NAMESPACE = 0                       # articles only
API_URL = "https://en.wikipedia.org/w/api.php"
API_BATCH = 50                          # revisions per API call (the MediaWiki limit for content)
API_PAUSE = 0.5                         # seconds between calls; Wikimedia asks for serial, polite clients
RAG_MAX_TITLES = 1000                   # cap on tracked articles: keeps API calls, embeddings and RAM small

# ---- chunking and embeddings ----
MIN_CHUNK_WORDS = 8
MAX_CHUNK_WORDS = 220
EMBED_DIM = 256                         # WordLlama l2_supercat 256-d
SEED = 7
