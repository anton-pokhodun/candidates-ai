"""
Agent-safe Candidate Search Tools
---------------------------------
Fully rewritten to:
- Deduplicate candidates
- Normalize Chroma scores safely
- Print raw + normalized scores
- Keep JSON output intact
"""

from typing import Optional, List, Dict, Any
from db_utils import get_vector_index


# ---------------------------------------------------------
# Score Normalization (Chroma = cosine similarity 0..1)
# ---------------------------------------------------------
def normalize_chroma_score(score: float) -> float:
    """Normalize Chroma cosine similarity score to 0..1."""
    if score is None:
        score = 0.0
    normalized = max(0.0, min(1.0, float(score)))
    print(f"Raw Chroma score: {score}, Normalized: {normalized}")
    return normalized


# ---------------------------------------------------------
# Safe Content Truncation
# ---------------------------------------------------------
def _truncate(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "... [truncated]"


# ---------------------------------------------------------
# Deduplication & Normalization
# ---------------------------------------------------------
def _dedupe_and_sort(results, top_k: int, min_score: float = 0.1):
    seen = set()
    unique = []

    for r in results:
        cid = r.node.metadata.get("candidate_id")
        norm_score = max(0.0, min(1.0, float(r.score) if r.score else 0.0))
        if cid is None or cid in seen:
            continue
        # if norm_score < min_score:
        #     print(f"Skipping candidate {cid} below threshold: {norm_score:.2f}")
        #     continue

        seen.add(cid)
        print(f"Keeping candidate {cid}, normalized score: {norm_score:.3f}")
        unique.append({"node_with_score": r, "normalized_score": norm_score})

    unique.sort(key=lambda x: x["normalized_score"], reverse=True)
    return unique[:top_k]


# ---------------------------------------------------------
# Serialize candidates for JSON output
# ---------------------------------------------------------
def _serialize_candidates(filtered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []
    for rdict in filtered:
        r = rdict["node_with_score"]
        norm_score = rdict["normalized_score"]
        node = r.node
        meta = node.metadata or {}

        serialized.append(
            {
                "candidate_id": meta.get("candidate_id"),
                "candidate_name": meta.get("candidate_name", "Unknown"),
                "file_name": meta.get("file_name", "unknown"),
                "score_raw": r.score,
                "score": norm_score,
                "content": _truncate(node.get_content() or ""),
            }
        )
    return serialized


# ---------------------------------------------------------
# Main Search (Human-readable summary)
# ---------------------------------------------------------
def search_candidates(query: str, top_k: int = 50) -> dict:
    try:
        index = get_vector_index()
        retriever = index.as_retriever(similarity_top_k=top_k * 5)
        results = retriever.retrieve(query)
    except Exception as e:
        return {"success": False, "error": f"Search failed: {str(e)}"}

    if not results:
        return {"success": True, "text": "No candidates found.", "candidates": []}

    filtered = _dedupe_and_sort(results, top_k)
    print(f"Returning {len(filtered)} unique candidates after deduplication.")

    # Build short summary
    lines = []
    for idx, rdict in enumerate(filtered, 1):
        r = rdict["node_with_score"]
        norm_score = rdict["normalized_score"]
        meta = r.node.metadata or {}
        lines.append(
            f"{idx}. {meta.get('candidate_name', 'Unknown')} "
            f"(ID: {meta.get('candidate_id', 'N/A')}, Score: {norm_score:.3f})"
        )

    return {
        "success": True,
        "query": query,
        "summary": "\n".join(lines),
        "candidates": _serialize_candidates(filtered),
    }
