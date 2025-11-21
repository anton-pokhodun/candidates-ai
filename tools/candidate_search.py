"""
Candidate Search Pipeline
Pure fuzzy matching — no aliases.
Profession + skills + semantic fallback.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore
from rapidfuzz import fuzz

from db_utils import (
    get_llm,
    get_vector_index,
    load_existing_metadata,
)

PROFESSION_THRESHOLD = 0.7
SKILL_THRESHOLD = 0.5
FINAL_THRESHOLD = 0.7

# ---------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------


@dataclass
class Candidate:
    candidate_id: int
    candidate_name: str
    profession: Optional[str]
    skills: List[str] = field(default_factory=list)
    file_name: Optional[str] = None


@dataclass
class SkillMatchResult:
    candidate_id: int
    candidate_name: str
    profession: Optional[str]
    skills: List[str]
    skill_score: float


@dataclass
class ProfessionMatchResult:
    candidate_id: int
    candidate_name: str
    profession: Optional[str]
    profession_score: float


@dataclass
class SearchResult:
    candidate_id: int
    candidate_name: str
    profession: Optional[str]
    skills: List[str]
    profession_score: float
    skill_score: float
    final_score: float


# ---------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------


def normalize_chroma_score(score: Optional[float]) -> float:
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
# Load metadata as Candidate objects
# ---------------------------------------------------------


def load_candidates() -> Dict[int, Candidate]:
    raw = load_existing_metadata()
    converted = {}

    for _file, meta in raw.items():
        cid = meta.get("candidate_id")
        if cid is None:
            continue

        converted[cid] = Candidate(
            candidate_id=cid,
            candidate_name=meta.get("candidate_name", "unknown"),
            profession=meta.get("profession"),
            skills=meta.get("skills", []) or [],
            file_name=meta.get("file_name"),
        )

    return converted


# ---------------------------------------------------------
# LLM QUERY PARSING
# ---------------------------------------------------------


def parse_query_with_llm(query: str) -> dict:
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
    except Exception:
        return {"profession": None, "skills": []}

    return {
        "profession": parsed.get("profession"),
        "skills": parsed.get("skills") or [],
    }


# ---------------------------------------------------------
# SKILL MATCHING
# ---------------------------------------------------------


def clean_skill(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_match_two_skills(q: str, c: str) -> float:
    qn = clean_skill(q)
    cn = clean_skill(c)
    if not qn or not cn:
        return 0
    return fuzz.token_sort_ratio(qn, cn)


def best_skill_match(
    query_skill: str, candidate_skills: List[str], threshold: int = 80
):
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
    if not query_skills:
        return 0.0

    used = set()
    matched = 0

    for q in query_skills:
        best, score = best_skill_match(q, candidate_skills, threshold)
        if best and best not in used:
            used.add(best)
            matched += 1

    return matched / len(query_skills)


def match_skills(
    query_skills: List[str], candidates: Dict[int, Candidate]
) -> Dict[int, SkillMatchResult]:
    results = {}

    for cid, cand in candidates.items():
        score = skill_overlap_score(query_skills, cand.skills)

        if score >= SKILL_THRESHOLD:
            results[cid] = SkillMatchResult(
                candidate_id=cand.candidate_id,
                candidate_name=cand.candidate_name,
                profession=cand.profession,
                skills=cand.skills,
                skill_score=score,
            )

    return results


# ---------------------------------------------------------
# PROFESSION MATCH
# ---------------------------------------------------------


def fuzzy_match_profession(
    query_prof: str, candidates: Dict[int, Candidate]
) -> Dict[int, ProfessionMatchResult]:
    results = {}
    qp = query_prof.lower().strip()

    for cid, cand in candidates.items():
        cp = (cand.profession or "").lower()
        score = fuzz.partial_ratio(qp, cp) / 100.0

        results[cid] = ProfessionMatchResult(
            candidate_id=cand.candidate_id,
            candidate_name=cand.candidate_name,
            profession=cand.profession,
            profession_score=score,
        )

    return results


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------


def semantic_search(query: str, top_k: int = 20):
    index = get_vector_index()
    retriever = index.as_retriever(similarity_top_k=top_k * 5)

    results: List[NodeWithScore] = retriever.retrieve(query)
    if not results:
        return []

    seen = set()
    unique = []

    for r in results:
        cid = r.node.metadata.get("candidate_id")
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(r)

    unique = sorted(unique, key=lambda r: normalize_chroma_score(r.score), reverse=True)
    unique = unique[:top_k]

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
# MAIN SEARCH LOGIC
# ---------------------------------------------------------


def search_candidates(query: str, top_k: int = 20) -> dict:
    print(f"🔍 New search query: {query}")

    parsed = parse_query_with_llm(query)
    profession = parsed.get("profession")
    skills = parsed.get("skills", [])

    print(f"Parsed → Profession: {profession}, Skills: {skills}")

    candidates = load_candidates()
    ranked = []

    # -----------------------------
    # Profession + Skills
    # -----------------------------
    if profession and skills:
        print("Matching candidates by profession + skills...")

        prof_matches = fuzzy_match_profession(profession, candidates)
        skill_matches = match_skills(skills, candidates)

        for cid, cand in candidates.items():
            c = prof_matches.get(cid)
            prof_score = c.profession_score if c is not None else 0.0
            s = skill_matches.get(cid)
            skill_score = s.skill_score if s is not None else 0.0

            # Weighted final score. Profession is 70%, skills 30%
            final = 0.7 * prof_score + 0.3 * skill_score

            print(
                f"Candidate ID: {cid}, Prof: {prof_score}, Skills: {skill_score}, Final: {final}"
            )

            if final > FINAL_THRESHOLD:
                ranked.append(
                    SearchResult(
                        candidate_id=cand.candidate_id,
                        candidate_name=cand.candidate_name,
                        profession=cand.profession,
                        skills=cand.skills,
                        profession_score=prof_score,
                        skill_score=skill_score,
                        final_score=final,
                    )
                )

    # -----------------------------
    # Only profession
    # -----------------------------
    elif profession:
        print("Matching candidates by profession...")

        prof_matches = fuzzy_match_profession(profession, candidates)

        for cid, cand in candidates.items():
            c = prof_matches.get(cid)
            ps = c.profession_score if c is not None else 0.0
            print(f"Candidate ID: {cid}, Prof Score: {ps}")

            if ps > PROFESSION_THRESHOLD:
                ranked.append(
                    SearchResult(
                        candidate_id=cand.candidate_id,
                        candidate_name=cand.candidate_name,
                        profession=cand.profession,
                        skills=cand.skills,
                        profession_score=ps,
                        skill_score=0.0,
                        final_score=ps,
                    )
                )

    # -----------------------------
    # Only skills
    # -----------------------------
    elif skills:
        print("Matching candidates by skills...")

        skill_matches = match_skills(skills, candidates)

        for cid, cand in candidates.items():
            s = skill_matches.get(cid)
            ss = s.skill_score if s is not None else 0.0
            print(f"Candidate ID: {cid}, Skill Score: {ss}")

            if ss > 0:
                ranked.append(
                    SearchResult(
                        candidate_id=cand.candidate_id,
                        candidate_name=cand.candidate_name,
                        profession=cand.profession,
                        skills=cand.skills,
                        profession_score=0.0,
                        skill_score=ss,
                        final_score=ss,
                    )
                )

    # -----------------------------
    # Neither → semantic fallback
    # -----------------------------
    else:
        print("⚠️ No profession or skills — semantic fallback...")
        return {
            "success": True,
            "final": True,
            "candidates": semantic_search(query, top_k),
        }

    ranked.sort(key=lambda x: x.final_score, reverse=True)

    return {
        "success": True,
        "final": True,
        "candidates": [r.__dict__ for r in ranked[:top_k]],
    }
