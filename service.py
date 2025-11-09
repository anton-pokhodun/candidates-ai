from llama_index.core import VectorStoreIndex, PromptTemplate
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.openai import OpenAI

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


def get_all_candidates():
    """Retrieve all unique candidate names from the ChromaDB collection."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    # Get all documents from the collection
    all_docs = collection.get(include=["metadatas"])

    # Extract unique candidate names
    candidate_names = set()
    candidate_info = {}

    for metadata in all_docs.get("metadatas", []):
        if metadata and "candidate_name" in metadata:
            name = metadata["candidate_name"]
            candidate_names.add(name)

            # Store additional info for each candidate
            if name not in candidate_info:
                candidate_info[name] = {
                    "name": name,
                    "file_name": metadata.get("file_name", "unknown"),
                    # "profession": metadata.get("profession", "Unknown"),
                }

    return sorted(list(candidate_names)), candidate_info


def get_candidate_by_id(candidate_id: str, use_llm: bool = True):
    """Retrieve detailed information for a specific candidate by ID."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    # Get all documents from the collection
    all_docs = collection.get(include=["metadatas", "documents"])

    # Find all chunks belonging to this candidate
    candidate_chunks = []
    candidate_metadata = None

    metadatas = all_docs.get("metadatas", [])
    documents = all_docs.get("documents", [])

    for i, metadata in enumerate(metadatas):
        if metadata and metadata.get("candidate_name") == candidate_id:
            if candidate_metadata is None:
                candidate_metadata = {
                    "name": metadata.get("candidate_name"),
                    "file_name": metadata.get("file_name", "unknown"),
                    "profession": metadata.get("profession", "Unknown"),
                }

            if i < len(documents):
                candidate_chunks.append(documents[i])

    if candidate_metadata is None:
        return None

    # Combine all chunks to get full CV text
    full_text = "\n\n".join(candidate_chunks)

    result = {
        **candidate_metadata,
        "chunks_count": len(candidate_chunks),
        "full_text": full_text,
        "chunks": candidate_chunks,
    }

    return result


def generate_candidate_summary_stream(candidate_id: str):
    """Generate a streaming response for candidate summary."""
    candidate_data = get_candidate_by_id(candidate_id, use_llm=False)

    if candidate_data is None:
        yield 'data: {"error": "Candidate not found"}\n\n'
        return

    # First, send the basic metadata
    import json

    metadata = {
        "name": candidate_data["name"],
        "file_name": candidate_data["file_name"],
        "profession": candidate_data["profession"],
    }
    yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

    # Then stream the LLM response
    full_text = candidate_data["full_text"]
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = f"""Based on the following CV information, create a well-structured professional summary.

CV Content:
{full_text}

Please provide a comprehensive summary with the following sections:
1. **Full Name**: Extract the candidate's full name
2. **Current Position**: Current or most recent job title and company
3. **Professional Summary**: A brief 2-3 sentence overview of their career
4. **Years of Experience**: Calculate total years of professional experience based on employment dates
5. **Key Skills**: List all technical skills, tools, frameworks, and technologies (organized by category if applicable)
6. **Work Experience**: Summarize each position with company name, role, dates, and key responsibilities/achievements
7. **Education**: Degrees, institutions, and graduation years
8. **Certifications**: Any professional certifications or additional training
9. **Notable Achievements**: Key accomplishments or projects worth highlighting

Format the response in a clear, professional manner using markdown. Be concise but thorough.
If any information is not available in the CV, indicate "Not specified" for that section.
"""

    # Stream the response
    response_stream = llm.stream_complete(prompt)

    for chunk in response_stream:
        if chunk.delta:
            yield f"data: {json.dumps({'type': 'content', 'data': chunk.delta})}\n\n"

    yield 'data: {"type": "done"}\n\n'


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
    candidate_id = "Candidate#1"  # Change this to any candidate ID from the list above
    candidate = get_candidate_by_id(candidate_id, use_llm=True)

    if candidate:
        print(f"\n{'=' * 80}")
        print(f"Candidate: {candidate['name']}")
        print(f"File: {candidate['file_name']}")
        print(f"Chunks: {candidate['chunks_count']}")
        print(f"{'=' * 80}\n")

        # Print the LLM-formatted summary
        if "formatted_summary" in candidate:
            print(candidate["formatted_summary"])
        else:
            print("No formatted summary available (use_llm=False)")
            print("\nRaw CV Text:")
            print(candidate["full_text"])
    else:
        print(f"\nCandidate '{candidate_id}' not found!")

    print("\n" + "=" * 80)
