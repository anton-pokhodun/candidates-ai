"""Candidate search tools for the agent."""

from typing import Optional
from db_utils import get_vector_index


def search_candidates(query: str, top_k: int = 50) -> str:
    """Search for candidates based on a natural language query.

    Args:
        query: Natural language search query
        top_k: Number of top results to return

    Returns:
        Formatted string containing candidate information
    """
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    if not results:
        return "No candidates found matching your query."

    return _format_search_results(results)


def search_candidates_structured(query: str, top_k: Optional[int] = 10) -> dict:
    """Search for candidates and return structured data for API.

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        Dictionary with answer and candidates list
    """
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    candidates = [
        {
            "candidate_id": str(result.node.metadata.get("candidate_id", "N/A")),
            "candidate_name": result.node.metadata.get("candidate_name", "Unknown"),
            "file_name": result.node.metadata.get("file_name", "unknown"),
            "score": float(result.score) if result.score is not None else 0.0,
            "content": result.node.get_content(),
        }
        for result in results
    ]

    return {
        "answer": f"Found {len(candidates)} candidates matching your query.",
        "candidates": candidates,
    }


def _format_search_results(results) -> str:
    """Format search results into a readable string.

    Args:
        results: Search results from the retriever

    Returns:
        Formatted string with candidate details
    """
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
