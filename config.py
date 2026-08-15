"""
Central configuration. Change models/paths here — nowhere else.
"""

import os

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.1

# --- Storage ---
PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "financial_reports")

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval ---
DEFAULT_TOP_K = 4
# Chunks with a relevance score below this are dropped rather than sent to the LLM.
# Chroma's default distance metric is cosine distance (lower = more similar), so this
# is a MAX distance, not a similarity score. Tune this against your own data —
# it depends heavily on document type and embedding model.
MAX_RELEVANCE_DISTANCE = 0.5

# --- Limits ---
MAX_FILE_SIZE_MB = 50
MAX_FILES_PER_BATCH = 20
