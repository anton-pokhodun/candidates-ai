import asyncio
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.api import ClientAPI

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
COLLECTION_NAME = "csv"
CHROMA_DB_PATH = "./chroma_db"
LLM_MODEL = "gpt-4o-mini"


# ============================================================================
# Database Client
# ============================================================================
def get_chroma_client() -> ClientAPI:
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


# ============================================================================
# Tool 1: Candidate Retrieval Tool
# ============================================================================
def search_candidates(query: str, top_k: int = 3) -> str:
    """Search for candidates based on a natural language query.

    This tool searches through candidate resumes and returns relevant information
    about candidates matching the query criteria.

    Args:
        query: Natural language search query (e.g., "Python developer with 5 years experience")
        top_k: Number of top results to return (default: 3)

    Returns:
        A formatted string containing candidate information matching the query
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Initialize embedding model
    embed_model = OpenAIEmbedding()

    # Create vector store and index
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    # Perform similarity search
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    if not results:
        return "No candidates found matching your query."

    # Format results with metadata
    formatted_results = []
    for idx, result in enumerate(results, 1):
        candidate_name = result.node.metadata.get("candidate_name", "Unknown")
        candidate_id = result.node.metadata.get("candidate_id", "N/A")
        file_name = result.node.metadata.get("file_name", "unknown")
        score = result.score
        text = result.node.get_content()

        formatted_results.append(
            f"Result {idx}:\n"
            f"Candidate: {candidate_name}\n"
            f"ID: {candidate_id}\n"
            f"File: {file_name}\n"
            f"Relevance Score: {score:.4f}\n"
            f"Content:\n{text}\n"
            f"{'-' * 60}"
        )

    return "\n\n".join(formatted_results)


def search_candidates_structured(query: str, top_k: Optional[int] = 10) -> dict:
    """Search for candidates and return structured data for API."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    embed_model = OpenAIEmbedding()
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    candidates = []
    for result in results:
        candidates.append(
            {
                "candidate_id": str(result.node.metadata.get("candidate_id", "N/A")),
                "candidate_name": result.node.metadata.get("candidate_name", "Unknown"),
                "file_name": result.node.metadata.get("file_name", "unknown"),
                "score": float(result.score),
                "content": result.node.get_content(),
            }
        )

    return {
        "answer": f"Found {len(candidates)} candidates matching your query.",
        "candidates": candidates,
    }


# ============================================================================
# Agent Setup
# ============================================================================
def create_agent() -> ReActAgent:
    """Create and configure the ReAct agent with all tools."""

    # Initialize LLM
    llm = OpenAI(model=LLM_MODEL, temperature=0.1)
    # Create tools
    tools = [
        FunctionTool.from_defaults(
            fn=search_candidates,
            name="search_candidates",
            description="Search for candidates using natural language queries. Use this to find candidates with specific skills, experience, or qualifications.",
        ),
    ]

    # Create agent
    agent = ReActAgent(tools=tools, llm=llm, verbose=True)
    return agent


async def search_with_agent(query: str, top_k: int = 10):
    """Use ReAct agent to search for candidates and return structured results."""
    agent = create_agent()
    ctx = Context(agent)
    print(f"Processing query: {query}")
    # Get structured results first
    structured_results = search_candidates_structured(query, top_k)

    # Send metadata first
    yield f"data: {json.dumps({'type': 'metadata', 'data': {'candidates': structured_results['candidates']}})}\n\n"

    # Run agent and stream response
    handler = agent.run(query, ctx=ctx)

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta)
            chunk_data = {"type": "content", "data": ev.delta}
            yield f"data: {json.dumps(chunk_data)}\n\n"

    await handler

    # Send completion signal
    yield f"data: {json.dumps({'type': 'done'})}\n\n"
