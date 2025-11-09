from chromadb.api import ClientAPI
from dotenv import load_dotenv
import chromadb
from llama_index.llms.openai import OpenAI
from typing import Dict, Optional, Generator, Any
import json

load_dotenv()

COLLECTION_NAME = "csv"
CHROMA_DB_PATH = "./chroma_db"


# ============================================================================
# Database Client
# ============================================================================
def get_chroma_client() -> ClientAPI:
    """Get or create a persistent ChromaDB client.

    Returns:
        chromadb.PersistentClient: Initialized ChromaDB client
    """
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


# ============================================================================
# Candidate Retrieval
# ============================================================================
def get_all_candidates() -> Dict[int, Dict[str, Any]]:
    """Retrieve all unique candidates from the ChromaDB collection.

    Returns:
        Dict mapping candidate_id to candidate information containing:
            - candidate_id: Unique identifier
            - candidate_name: Full name
            - file_name: Source file name
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    all_docs = collection.get(include=["metadatas"])

    candidates = {}
    for metadata in all_docs.get("metadatas") or []:
        if not metadata or "candidate_id" not in metadata:
            continue

        candidate_id = metadata["candidate_id"]
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "candidate_name": metadata.get("candidate_name"),
            "file_name": metadata.get("file_name", "unknown"),
        }

    return candidates


def get_candidate_by_id(candidate_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve detailed information for a specific candidate by ID.

    Args:
        candidate_id: Unique identifier for the candidate

    Returns:
        Dictionary containing candidate information:
            - candidate_id: Unique identifier
            - candidate_name: Full name
            - file_name: Source file name
        Returns None if candidate not found
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    all_docs = collection.get(include=["metadatas", "documents"])

    metadatas = all_docs.get("metadatas", [])
    documents = all_docs.get("documents", [])

    candidate_chunks = []
    candidate_metadata = None

    for i, metadata in enumerate(metadatas or []):
        if not metadata or metadata.get("candidate_id") != int(candidate_id):
            continue

        # Store metadata from first matching chunk
        if candidate_metadata is None:
            candidate_metadata = {
                "candidate_id": metadata.get("candidate_id"),
                "candidate_name": metadata.get("candidate_name"),
                "file_name": metadata.get("file_name", "unknown"),
            }

        # Collect all text chunks for this candidate
        if documents and i < len(documents):
            candidate_chunks.append(documents[i])

    if candidate_metadata is None:
        return None

    return {
        **candidate_metadata,
        "chunks_count": len(candidate_chunks),
        "full_text": "\n\n".join(candidate_chunks),
        "chunks": candidate_chunks,
    }


# ============================================================================
# Summary Generation
# ============================================================================
def _build_summary_prompt(cv_text: str) -> str:
    """Build the prompt for CV summary generation.

    Args:
        cv_text: Complete CV text

    Returns:
        Formatted prompt string
    """
    return f"""Based on the following CV information, create a well-structured professional summary.

CV Content:
{cv_text}

Please provide a comprehensive summary with the following sections:
1. **Current Position**: Current or most recent job title and company
2. **Professional Summary**: A brief 2-3 sentence overview of their career
3. **Years of Experience**: Calculate total years of professional experience based on employment dates
4. **Key Skills**: List all technical skills, tools, frameworks, and technologies (organized by category if applicable)
5. **Work Experience**: Summarize each position with company name, role, dates, and key responsibilities/achievements
6. **Education**: Degrees, institutions, and graduation years
7. **Certifications**: Any professional certifications or additional training
8. **Notable Achievements**: Key accomplishments or projects worth highlighting

Format the response in a clear, professional manner using markdown. Be concise but thorough.
If any information is not available in the CV, indicate "Not specified" for that section.
"""


def generate_candidate_summary_stream(candidate_id: str) -> Generator[str, None, None]:
    """Generate a streaming Server-Sent Events response for candidate summary.

    Args:
        candidate_id: Unique identifier for the candidate

    Yields:
        SSE-formatted strings containing:
            - metadata event with candidate information
            - content events with streaming summary text
            - done event when complete
            - error event if candidate not found
    """
    candidate_data = get_candidate_by_id(candidate_id)

    if candidate_data is None:
        yield 'data: {"error": "Candidate not found"}\n\n'
        return

    # Send candidate metadata
    metadata = {
        "candidate_id": candidate_data["candidate_id"],
        "candidate_name": candidate_data["candidate_name"],
        "file_name": candidate_data["file_name"],
    }
    yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"

    # Initialize LLM and generate summary
    llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = _build_summary_prompt(candidate_data["full_text"])

    # Stream the LLM response
    response_stream = llm.stream_complete(prompt)
    for chunk in response_stream:
        if chunk.delta:
            yield f"data: {json.dumps({'type': 'content', 'data': chunk.delta})}\n\n"

    yield 'data: {"type": "done"}\n\n'
