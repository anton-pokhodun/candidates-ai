"""Database utilities for ChromaDB operations."""

import chromadb
from chromadb.api import ClientAPI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

from config import CHROMA_DB_PATH, COLLECTION_NAME


def get_chroma_client() -> ClientAPI:
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_vector_index(collection_name: str = COLLECTION_NAME) -> VectorStoreIndex:
    """Create a vector store index from ChromaDB collection.

    Args:
        collection_name: Name of the ChromaDB collection

    Returns:
        VectorStoreIndex configured with the collection
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    embed_model = OpenAIEmbedding()
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
