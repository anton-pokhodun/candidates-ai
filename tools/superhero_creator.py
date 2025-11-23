"""
SAFE Superhero Candidate Creator Tool
-------------------------------------
This tool MUST only be called when the user EXPLICITLY asks to create,
merge, blend, or combine candidates into a superhero.

It cannot be triggered from search tasks.
"""

import random
from typing import List, Dict, Union
from dataclasses import dataclass
from llama_index.llms.openai import OpenAI

from db_utils import get_chroma_client
from config import COLLECTION_NAME, LLM_MODEL


# --- Constants ---

SUPERHERO_MIDDLE_NAMES = [
    "Dragon",
    "Beast",
    "Rock",
    "Thunder",
    "Storm",
    "Steel",
    "Phoenix",
    "Titan",
    "Viper",
    "Shadow",
    "Blaze",
    "Frost",
    "Venom",
    "Raven",
    "Wolf",
    "Hawk",
    "Cobra",
    "Tiger",
]


# --- Dataclasses ---


@dataclass
class CandidateData:
    name: str
    content: str
    MAX_CHARS: int = 3000


@dataclass
class Superhero:
    name: str
    source_candidates: List[CandidateData]
    profile: str


# --- PUBLIC ENTRY POINT (SAFE) ---


def build_superhero(candidate_names: str, *, force_superhero: bool = False) -> str:
    """
    SAFE ENTRY POINT.

    This function will ONLY run if force_superhero=True.
    The agent layer should set force_superhero=True ONLY when the user
    explicitly asks to create a superhero.

    This prevents accidental tool invocation during candidate searches.
    """

    if not force_superhero:
        return (
            "Error: Superhero creation was attempted without explicit user request. "
            "This tool must only run when the user explicitly asks to create a superhero."
        )

    try:
        names = [n.strip() for n in candidate_names.split(",") if n.strip()]

        if not (2 <= len(names) <= 3):
            return "Error: Please provide exactly 2 or 3 candidate names."

        candidates_data = _retrieve_candidates(names)

        if isinstance(candidates_data, str):  # error string
            return candidates_data

        superhero_name = _make_superhero_name(candidates_data)
        profile = _make_superhero_profile(candidates_data, superhero_name)

        return _format_superhero_output(
            Superhero(
                name=superhero_name, source_candidates=candidates_data, profile=profile
            )
        )

    except Exception as e:
        return f"Error creating superhero: {e}"


# --- Helpers ---


def _retrieve_candidates(names: List[str]) -> Union[List[CandidateData], str]:
    client = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME)

    results_accumulated: List[CandidateData] = []

    for name in names:
        res = collection.get(where={"candidate_name": name}, include=["documents"])

        if not res["documents"]:
            return f"Error: Candidate '{name}' not found."

        full = " ".join(res["documents"])
        max_chars = CandidateData.MAX_CHARS
        truncated = full[:max_chars] + (
            "... [truncated]" if len(full) > max_chars else ""
        )

        results_accumulated.append(CandidateData(name=name, content=truncated))

    return results_accumulated


def _make_superhero_name(candidates: List[CandidateData]) -> str:
    """Generate a superhero name."""
    first_parts = candidates[0].name.split()
    second_parts = candidates[1].name.split()

    first = first_parts[0] if first_parts else "Super"
    last = second_parts[-1] if second_parts else "Hero"

    middle = random.choice(SUPERHERO_MIDDLE_NAMES)
    return f"{first} '{middle}' {last}"


def _make_superhero_profile(candidates: List[CandidateData], name: str) -> str:
    """Generate superhero profile body text (NO TOOL LOGIC IN PROMPT!)."""

    llm = OpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=1000,
        timeout=30.0,
    )

    candidate_bodies = "\n\n".join(
        f"Candidate {i + 1} — {c.name}:\n{c.content}" for i, c in enumerate(candidates)
    )

    prompt = f"""
You are combining the following candidates into a single elite 'superhero' résumé.

Superhero Name: {name}

Combine their strongest:
- skills
- experiences
- technologies
- achievements

Remove duplicates. Organize clearly.
Keep concise (under 800 words).

Candidate Data:
{candidate_bodies}

Now produce the final merged profile:
"""

    return llm.complete(prompt).text


def _format_superhero_output(hero: Superhero) -> str:
    names = ", ".join(c.name for c in hero.source_candidates)
    return f"""
🦸 SUPERHERO CREATED 🦸

Name: {hero.name}
Combined from {len(hero.source_candidates)} candidates:
- {names}

--------------------------------------------------------------------------------

{hero.profile}

--------------------------------------------------------------------------------
"""
