"""
Agent-safe Candidate Search Tools
---------------------------------
Enhanced with:
- LLM-based query parsing
- Profession fuzzy matching
- Skillset scoring
- Combined ranking
- Semantic fallback
"""

import json
from typing import Any, Dict, List

from llama_index.core.llms import ChatMessage
from rapidfuzz import fuzz

from db_utils import get_llm, get_vector_index, load_existing_metadata


# ---------------------------------------------------------
# Score Normalization (Chroma = cosine similarity 0..1)
# ---------------------------------------------------------
def normalize_chroma_score(score: float) -> float:
    if score is None:
        score = 0.0
    return max(0.0, min(1.0, float(score)))


# ---------------------------------------------------------
# Safe Content Truncation
# ---------------------------------------------------------
def _truncate(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "... [truncated]"


# ---------------------------------------------------------
# Deduplication & Normalization for semantic search fallback
# ---------------------------------------------------------
def _dedupe_and_sort(results, top_k: int):
    seen = set()
    unique = []

    for r in results:
        cid = r.node.metadata.get("candidate_id")
        norm_score = normalize_chroma_score(r.score)

        if cid is None or cid in seen:
            continue

        seen.add(cid)
        unique.append({"node_with_score": r, "normalized_score": norm_score})

    unique.sort(key=lambda x: x["normalized_score"], reverse=True)
    return unique[:top_k]


# ---------------------------------------------------------
# Serialization for output
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
# LLM Query parsing
# ---------------------------------------------------------
def parse_query_with_llm(query: str) -> dict:
    """
    Returns JSON:
    {
        "profession": "history teacher",
        "skills": ["curriculum planning", "classroom management"]
    }
    """
    llm = get_llm()
    prompt = f"""
    Extract a profession and a list of skills from the query below.
    Return ONLY valid JSON:
    {{
        "profession": string or null,
        "skills": list of strings
    }}

    If no profession — return null.
    If no skills — return [].

    Query:
    {query}

    JSON:
    """
    messages = [ChatMessage(role="user", content=prompt)]
    response = llm.chat(messages)

    print(f"LLM extraction: {response.message.content}")

    try:
        if response and response.message and response.message.content:
            return json.loads(response.message.content)
        return {"profession": None, "skills": []}
    except Exception:
        return {"profession": None, "skills": []}


# ---------------------------------------------------------
# Profession fuzzy matching
# ---------------------------------------------------------
def fuzzy_match_profession(target_profession: str, metadata: dict):
    results = []
    for file, meta in metadata.items():
        candidate_prof = meta.get("profession", "").lower()
        score = fuzz.partial_ratio(target_profession.lower(), candidate_prof)

        results.append(
            {
                "file": file,
                "candidate_id": meta.get("candidate_id"),
                "candidate_name": meta.get("candidate_name"),
                "profession": meta.get("profession"),
                "skills": meta.get("skills", []),
                "profession_score": score,
            }
        )

    return sorted(results, key=lambda x: x["profession_score"], reverse=True)


# ---------------------------------------------------------
# Skill scoring
# ---------------------------------------------------------
def skill_overlap_score(query_skills, candidate_skills):
    if not query_skills:
        return 0.0
    qs = set(s.lower() for s in query_skills)
    cs = set(s.lower() for s in candidate_skills)
    overlap = qs.intersection(cs)
    return len(overlap) / len(qs)


# ---------------------------------------------------------
# Combine profession + skills into one score
# ---------------------------------------------------------
def rank_profession_and_skills(prof_matches, query_skills):
    ranked = []
    for m in prof_matches:
        skill_score = skill_overlap_score(query_skills, m["skills"])
        final = 0.7 * (m["profession_score"] / 100.0) + 0.3 * skill_score
        m["skill_score"] = skill_score
        m["final_score"] = final
        ranked.append(m)

    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)


# ---------------------------------------------------------
# Semantic fallback search
# ---------------------------------------------------------
def semantic_search(query: str, top_k: int = 20):
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k * 5)
    results = retriever.retrieve(query)

    if not results:
        return []

    filtered = _dedupe_and_sort(results, top_k)
    return _serialize_candidates(filtered)


# ---------------------------------------------------------
# MAIN SEARCH PIPELINE
# ---------------------------------------------------------
def search_candidates(query: str, top_k: int = 20) -> dict:
    print(f"🔍 New search query: {query}")

    parsed = parse_query_with_llm(query)
    profession = parsed["profession"]
    skills = parsed["skills"]

    print(f"Parsed → Profession: {profession}, Skills: {skills}")

    metadata = load_existing_metadata()

    # -----------------------------
    # Stage A: Profession Match
    # -----------------------------
    if profession:
        prof_matches = fuzzy_match_profession(profession, metadata)
        ranked = rank_profession_and_skills(prof_matches, skills)

        best_score = ranked[0]["final_score"] if ranked else 0
        print(f"Best profession/skill score = {best_score:.3f}")

        if best_score >= 0.60:
            return {
                "success": True,
                "method": "profession_skill_match",
                "query": query,
                "parsed": parsed,
                "candidates": ranked[:top_k],
            }

    # -----------------------------
    # Stage B: Semantic fallback
    # -----------------------------
    print("⚠️ Falling back to semantic search...")
    semantic = semantic_search(query, top_k=top_k)

    return {
        "success": True,
        "method": "semantic_fallback",
        "query": query,
        "parsed": parsed,
        "candidates": semantic,
    }
