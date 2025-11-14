"""Database utilities for ChromaDB operations."""

import chromadb
from chromadb.api import ClientAPI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL


def get_chroma_client() -> ClientAPI:
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_embedding_model() -> OpenAIEmbedding:
    """Get configured OpenAI embedding model.

    Returns:
        OpenAIEmbedding instance
    """
    return OpenAIEmbedding(model=EMBEDDING_MODEL)


def reset_collection(client: ClientAPI, collection_name: str) -> chromadb.Collection:
    """Delete existing collection and create a new one.

    Args:
        client: ChromaDB client instance
        collection_name: Name of the collection to reset

    Returns:
        chromadb.Collection: Newly created collection
    """
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except Exception:
        print(f"No existing collection '{collection_name}' to delete")

    collection = client.get_or_create_collection(name=collection_name)
    print(f"Created new collection '{collection_name}'")
    return collection


def get_vector_index(collection_name: str = COLLECTION_NAME) -> VectorStoreIndex:
    """Create a vector store index from ChromaDB collection.

    Args:
        collection_name: Name of the ChromaDB collection

    Returns:
        VectorStoreIndex configured with the collection
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    embed_model = get_embedding_model()
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
