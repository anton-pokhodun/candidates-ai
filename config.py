"""Configuration settings for the candidate search agent."""

# ChromaDB settings
COLLECTION_NAME = "csv"
CHROMA_DB_PATH = "./chroma_db"

# LLM settings
LLM_MODEL = "gpt-4o-mini"

# Document processing settings
DATA_DIR = "./data"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
