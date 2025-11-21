"""
Candidate Search Pipeline
Pure fuzzy matching — no aliases.
Profession + skills + semantic fallback.
"""

import json
import re
from typing import Dict, List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore
from rapidfuzz import fuzz

from db_utils import (
    get_llm,
    get_vector_index,
    load_existing_metadata,
)

# ---------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------


def normalize_chroma_score(score: Optional[float]) -> float:
    """Ensure vector score is between 0–1."""
    if score is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(score)))
    except (ValueError, TypeError):
        return 0.0


def truncate_text(text: str, max_chars: int = 300):
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + "…"


# ---------------------------------------------------------
# LLM QUERY PARSING
# ---------------------------------------------------------
def parse_query_with_llm(query: str) -> dict:
    """
    Extract profession + skills from user query.
    No hallucination into unknown keys.
    """
    llm = get_llm()

    prompt = f"""
Return ONLY JSON with this schema:

{{
  "profession": string or null,
  "skills": list of strings
}}

Rules:
If profession not mentioned → null. Do not provide a guess.
If no skills → [].

Query: "{query}"

JSON:
    """

    msg = [ChatMessage(role="user", content=prompt)]
    result = llm.chat(msg)

    if not result or not result.message or not result.message.content:
        return {"profession": None, "skills": []}

    try:
        parsed = json.loads(result.message.content)
        if not isinstance(parsed, dict):
            raise ValueError
    except json.JSONDecodeError:
        return {"profession": None, "skills": []}

    # Guarantee schema
    return {
        "profession": parsed.get("profession"),
        "skills": parsed.get("skills") or [],
    }


def clean_skill(s: str) -> str:
    """Basic cleaning. NO aliases, NO expansion."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\+]", " ", s)  # keep letters, numbers, +, _
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match_two_skills(q: str, c: str) -> float:
    """Fuzzy score between two normalized skill strings."""
    qn = clean_skill(q)
    cn = clean_skill(c)
    if not qn or not cn:
        return 0
    return fuzz.token_sort_ratio(qn, cn)


def best_skill_match(
    query_skill: str, candidate_skills: List[str], threshold: int = 80
):
    """Find the best fuzzy match for 1 query skill among candidate skills."""
    best = None
    best_score = 0

    for cand_skill in candidate_skills:
        score = fuzzy_match_two_skills(query_skill, cand_skill)
        if score > best_score:
            best_score = score
            best = cand_skill

    if best_score >= threshold:
        return best, best_score

    return None, best_score


def skill_overlap_score(
    query_skills: List[str], candidate_skills: List[str], threshold: int = 80
) -> float:
    """
    Query skill matched if fuzzy score ≥ threshold.
    No double-counting.
    """
    if not query_skills:
        return 0.0

    used = set()
    matched = 0

    for q in query_skills:
        best_match, score = best_skill_match(q, candidate_skills, threshold)
        if best_match and best_match not in used:
            used.add(best_match)
            matched += 1

    return matched / len(query_skills)


def match_skills(query_skills: List[str], metadata: Dict) -> Dict[int, Dict]:
    """Compute skill overlap score for all candidates."""
    results = {}

    for _file, meta in metadata.items():
        cid = meta.get("candidate_id")
        if cid is None:
            continue

        cskills = meta.get("skills", []) or []
        score = skill_overlap_score(query_skills, cskills)

        if score >= 0.5:
            results[cid] = {
                "candidate_id": cid,
                "candidate_name": meta.get("candidate_name"),
                "profession": meta.get("profession"),
                "skills": cskills,
                "skill_score": score,
            }

    return results


# ---------------------------------------------------------
# PROFESSION FUZZY MATCH
# ---------------------------------------------------------


def fuzzy_match_profession(query_prof: str, metadata: Dict) -> Dict[int, Dict]:
    results = {}

    qp = query_prof.lower().strip()

    for _file, meta in metadata.items():
        cid = meta.get("candidate_id")
        if cid is None:
            continue

        cp = (meta.get("profession") or "").lower()

        score = fuzz.partial_ratio(qp, cp)

        results[cid] = {
            "candidate_id": cid,
            "candidate_name": meta.get("candidate_name"),
            "profession": meta.get("profession"),
            "profession_score": score / 100.0,  # normalize 0–1
        }

    return results


# ---------------------------------------------------------
# COMBINE PROFESSION + SKILLS
# ---------------------------------------------------------


def combine_scores(prof_matches, skill_matches, metadata):
    combined = []

    for _file, meta in metadata.items():
        cid = meta.get("candidate_id")

        prof_score = prof_matches.get(cid, {}).get("profession_score", 0.0)
        skill_score = skill_matches.get(cid, {}).get("skill_score", 0.0)

        final = 0.70 * prof_score + 0.30 * skill_score

        combined.append(
            {
                "candidate_id": cid,
                "candidate_name": meta.get("candidate_name"),
                "profession": meta.get("profession"),
                "skills": meta.get("skills", []),
                "profession_score": prof_score,
                "skill_score": skill_score,
                "final_score": final,
            }
        )

    return sorted(combined, key=lambda x: x["final_score"], reverse=True)


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------


def semantic_search(query: str, top_k: int = 20):
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k * 5)

    results: List[NodeWithScore] = retriever.retrieve(query)
    if not results:
        return []

    # dedupe per candidate
    seen = set()
    unique = []

    for r in results:
        cid = r.node.metadata.get("candidate_id")
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(r)

    # sort by normalized vector score
    unique = sorted(unique, key=lambda r: normalize_chroma_score(r.score), reverse=True)
    unique = unique[:top_k]

    # serialize
    serialized = []
    for r in unique:
        meta = r.node.metadata
        serialized.append(
            {
                "candidate_id": meta.get("candidate_id"),
                "candidate_name": meta.get("candidate_name"),
                "profession": meta.get("profession"),
                "file_name": meta.get("file_name"),
                "score": normalize_chroma_score(r.score),
                "content": truncate_text(r.node.get_content() or ""),
            }
        )

    return serialized


# ---------------------------------------------------------
# MAIN ENTRY
# ---------------------------------------------------------


def search_candidates(query: str, top_k: int = 20) -> dict:
    print(f"🔍 New search query: {query}")

    parsed = parse_query_with_llm(query)
    profession = parsed.get("profession")
    skills = parsed.get("skills", [])

    print(f"Parsed → Profession: {profession}, Skills: {skills}")

    metadata = load_existing_metadata()

    ranked_candidates = []

    # -----------------------------
    # Both profession and skills defined
    # -----------------------------
    if profession and skills:
        print("Matching candidates by profession + skills...")
        prof_matches = fuzzy_match_profession(profession, metadata)
        skills_matches = match_skills(skills, metadata)
        for cid, meta in metadata.items():
            # Profession score normalized (0–1)
            candidate_id = meta.get("candidate_id", {})

            prof_entry = prof_matches.get(candidate_id)
            prof_score = float(prof_entry["profession_score"]) if prof_entry else 0.0

            skill_entry = skills_matches.get(candidate_id)
            skill_score = float(skill_entry["skill_score"]) if skill_entry else 0.0

            final_score = 0.7 * prof_score + 0.3 * skill_score
            print(
                f"Candidate ID: {candidate_id}, Prof Score: {prof_score}, Skill Score: {skill_score}, Final Score: {final_score}"
            )

            if final_score > 0.7:
                ranked_candidates.append(
                    {
                        "candidate_id": cid,
                        "candidate_name": meta.get("candidate_name"),
                        "profession": meta.get("profession"),
                        "skills": meta.get("skills", []),
                        "profession_score": prof_score,
                        "skill_score": skill_score,
                        "final_score": final_score,
                    }
                )

    # -----------------------------
    # Only profession defined
    # -----------------------------
    elif profession:
        print("Matching candidates by profession...")
        prof_matches = fuzzy_match_profession(profession, metadata)
        for cid, meta in metadata.items():
            candidate_id = meta.get("candidate_id")
            match_candidate = prof_matches.get(candidate_id) if candidate_id else None
            prof_score = float(
                (match_candidate["profession_score"]) if match_candidate else 0.0
            )
            print(f"Candidate ID: {candidate_id}, Prof Score: {prof_score}")
            if prof_score > 0.7:
                print(f"  → Matched with score {prof_score}")
                ranked_candidates.append(
                    {
                        **meta,
                        "prof_score": prof_score,
                        "skill_score": 0.0,
                        "final_score": prof_score,
                    }
                )

    # -----------------------------
    # Only skills defined
    # -----------------------------
    elif skills:
        print("Matching candidates by skills...")
        skills_matches = match_skills(skills, metadata)
        for cid, meta in metadata.items():
            candidate_id = meta.get("candidate_id")
            match_skill = skills_matches.get(candidate_id) if candidate_id else None
            skill_score = float(match_skill["skill_score"]) if match_skill else 0.0

            print(f"Candidate ID: {candidate_id}, Skill Score: {skill_score}")

            if skill_score > 0:
                print(f"  → Matched skill with score {skill_score}")
                ranked_candidates.append(
                    {
                        **meta,
                        "prof_score": 0.0,
                        "skill_score": skill_score,
                        "final_score": skill_score,
                    }
                )
    # -----------------------------
    # Neither defined → fallback
    # -----------------------------
    else:
        print("⚠️ No profession or skills found, using semantic search fallback...")
        semantic = semantic_search(query, top_k=top_k)
        return {
            "success": True,
            "final": True,
            "candidates": semantic,
        }

    ranked_candidates = [c for c in ranked_candidates if c["final_score"] > 0]
    print(
        f"Found {len(ranked_candidates)} candidates after profession + skills matching."
    )
    print(f"Ranking candidates by final score... {top_k} to return.")
    # -----------------------------
    # Sort by final score
    # -----------------------------
    return {
        "success": True,
        "candidates": ranked_candidates[:top_k],
        "final": True,
    }
