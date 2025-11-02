from llama_index.core import VectorStoreIndex, PromptTemplate
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

load_dotenv()

collection_name = "csv"


def get_chroma_client():
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path="./chroma_db")


def get_query_engine(
    similarity_top_k: int = 3, text_qa_template: PromptTemplate | None = None
):
    """Create a query engine from the persisted ChromaDB collection."""

    # Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
    # Get the persisted collection
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    # Create embedding model (must match the one used during indexing)
    embedding = OpenAIEmbedding()

    # Create vector store from existing collection
    vector_store = ChromaVectorStore(chroma_collection=collection, embedding=embedding)

    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embedding
    )

    # Return query engine
    return index.as_query_engine(
        similarity_top_k=similarity_top_k, text_qa_template=text_qa_template
    )


def print_retrieved_chunks(response):
    """Display retrieved chunks with scores and metadata."""
    print("\n" + "=" * 80)
    print(f"RETRIEVED CHUNKS ({len(response.source_nodes)} total)")
    print("=" * 80)

    for i, node in enumerate(response.source_nodes, 1):
        print(f"\n--- Chunk {i} (Score: {node.score:.4f}) ---")
        print(f"Text: {node.text}")
        if node.metadata:
            print(f"Metadata: {node.metadata}")


if __name__ == "__main__":
    # Example usage
    #
    qa_prompt_tmpl = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query in detail. Include specific information such as:\n"
        "- All Names and titles\n"
        "- Education and qualifications\n"
        "- Years of experience\n"
        "- Specific skills and areas of expertise\n"
        "- Notable achievements or projects\n"
        "- All certification, or diplomas if exist\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    query_engine = get_query_engine(
        similarity_top_k=10, text_qa_template=qa_prompt_tmpl
    )
    query = "I want to find a teacher of history. If there are any - provide me a summary of their background."

    response = query_engine.query(query)
    print(response)
