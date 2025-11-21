"""Superhero candidate creator tool - Refactored."""

import random
from typing import List, Dict, Union
from dataclasses import dataclass
from llama_index.llms.openai import OpenAI

# Assuming these are necessary for the retrieve function and configuration
from db_utils import get_chroma_client
from config import COLLECTION_NAME, LLM_MODEL

# --- Constants ---

# Removed MAX_CHARS_PER_CANDIDATE, MAX_RESPONSE_TOKENS, LLM_TIMEOUT
# as their values are now directly used/set in the functions

SUPERHERO_MIDDLE_NAMES: List[str] = [
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
    """Stores a single candidate's name and truncated content."""

    name: str
    content: str
    MAX_CHARS: int = 3000


@dataclass
class Superhero:
    """Stores the final generated superhero profile."""

    name: str
    source_candidates: List[CandidateData]
    profile: str


# --- Core Logic ---


def create_superhero(candidate_names: str) -> str:
    """
    Create a superhero candidate by combining skills from 2-3 candidates.
    """
    try:
        names = [name.strip() for name in candidate_names.split(",")]

        if not (2 <= len(names) <= 3):
            return "Error: Please provide 2 or 3 candidate names separated by commas."

        candidates_data: Union[List[CandidateData], str] = _retrieve_candidates(names)

        if isinstance(candidates_data, str):  # Error message
            return candidates_data

        superhero_name = _generate_superhero_name(candidates_data)
        profile = _generate_superhero_profile(candidates_data, superhero_name)

        superhero_result = Superhero(
            name=superhero_name, source_candidates=candidates_data, profile=profile
        )

        return _format_superhero_output(superhero_result)

    except Exception as e:
        return f"Error creating superhero: {str(e)}"


def _retrieve_candidates(names: List[str]) -> Union[List[CandidateData], str]:
    """Retrieve candidate data from ChromaDB and return as a list of CandidateData objects."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    candidates_data: List[CandidateData] = []

    for candidate_name in names:
        results = collection.get(
            where={"candidate_name": candidate_name},
            include=["documents"],
        )

        if not results["documents"]:
            return f"Error: Candidate with name '{candidate_name}' not found."

        full_content = " ".join(results["documents"])

        # Apply truncation logic using the dataclass attribute
        max_chars = CandidateData.MAX_CHARS
        truncated_content = full_content[:max_chars]
        if len(full_content) > max_chars:
            truncated_content += "... [truncated]"

        candidates_data.append(
            CandidateData(
                name=candidate_name,
                content=truncated_content,
            )
        )

    return candidates_data


def _generate_superhero_name(candidates_data: List[CandidateData]) -> str:
    """Generate a superhero name from candidate names."""

    # Safely get the first/last names from the first two candidates
    first_name_parts = candidates_data[0].name.split()
    last_name_parts = candidates_data[1].name.split()

    first_name = first_name_parts[0] if first_name_parts else "Super"
    last_name = (
        last_name_parts[-1]
        if len(last_name_parts) > 1
        else last_name_parts[0]
        if last_name_parts
        else "Hero"
    )

    middle_name = random.choice(SUPERHERO_MIDDLE_NAMES)
    return f"{first_name} '{middle_name}' {last_name}"


def _generate_superhero_profile(
    candidates_data: List[CandidateData], superhero_name: str
) -> str:
    """Generate superhero profile using LLM."""

    llm = OpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=1000,  # Used constant value
        timeout=30.0,  # Used constant value
    )

    candidates_info = "\n\n".join(
        [
            f"Candidate {i + 1} ({data.name}):\n{data.content}"
            for i, data in enumerate(candidates_data)
        ]
    )

    prompt = f"""You are creating a "superhero" candidate by combining the best skills and qualifications from multiple candidates.

Here are the candidates:

{candidates_info}

Task:
1. Extract the key skills, technologies, experiences, and qualifications from each candidate
2. Combine them into one comprehensive profile highlighting the BEST and most impressive aspects from each
3. Remove duplicates and organize by category (Technical Skills, Experience, Education, etc.)
4. Make it read like a powerful, combined resume profile
5. Keep the response concise and under 800 words

Superhero Name: {superhero_name}

Create a compelling superhero candidate profile:"""

    response = llm.complete(prompt)
    return response.text


def _format_superhero_output(superhero: Superhero) -> str:
    """Format the final Superhero output using the dataclass."""

    candidate_names = ", ".join([data.name for data in superhero.source_candidates])
    num_candidates = len(superhero.source_candidates)

    return f"""
🦸 SUPERHERO CANDIDATE CREATED! 🦸

Name: {superhero.name}
Combined from: {num_candidates} candidates
- {candidate_names}

{"-" * 80}

{superhero.profile}

{"-" * 80}

This superhero candidate combines the best qualities from all {num_candidates} candidates!
"""
