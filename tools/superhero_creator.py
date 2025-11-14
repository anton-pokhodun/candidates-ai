"""Superhero candidate creator tool."""

import random
from typing import List, Dict
from llama_index.llms.openai import OpenAI

from db_utils import get_chroma_client
from config import COLLECTION_NAME, LLM_MODEL


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

MAX_CHARS_PER_CANDIDATE = 3000
MAX_RESPONSE_TOKENS = 1000
LLM_TIMEOUT = 30.0


def create_superhero(candidate_names: str) -> str:
    """Create a superhero candidate by combining skills from 2-3 candidates.

    Args:
        candidate_names: Comma-separated candidate names

    Returns:
        Formatted string describing the superhero candidate
    """
    try:
        names = [name.strip() for name in candidate_names.split(",")]

        if len(names) < 2 or len(names) > 3:
            return "Error: Please provide 2 or 3 candidate names separated by commas."

        candidates_data = _retrieve_candidates(names)
        if isinstance(candidates_data, str):  # Error message
            return candidates_data

        superhero_name = _generate_superhero_name(candidates_data)
        profile = _generate_superhero_profile(candidates_data, superhero_name)

        return _format_superhero_output(superhero_name, candidates_data, profile)

    except Exception as e:
        return f"Error creating superhero: {str(e)}"


def _retrieve_candidates(names: List[str]) -> List[Dict[str, str]] | str:
    """Retrieve candidate data from ChromaDB.

    Args:
        names: List of candidate names

    Returns:
        List of candidate data dictionaries or error message
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    candidates_data = []
    for candidate_name in names:
        results = collection.get(
            where={"candidate_name": candidate_name},
            include=["documents", "metadatas"],
        )

        if not results["documents"]:
            return f"Error: Candidate with name '{candidate_name}' not found."

        full_content = " ".join(results["documents"])
        truncated_content = full_content[:MAX_CHARS_PER_CANDIDATE]
        if len(full_content) > MAX_CHARS_PER_CANDIDATE:
            truncated_content += "... [truncated]"

        candidates_data.append(
            {
                "name": candidate_name,
                "content": truncated_content,
            }
        )

    return candidates_data


def _generate_superhero_name(candidates_data: List[Dict[str, str]]) -> str:
    """Generate a superhero name from candidate names.

    Args:
        candidates_data: List of candidate data

    Returns:
        Generated superhero name
    """
    first_candidate_name = candidates_data[0]["name"].split()
    second_candidate_name = candidates_data[1]["name"].split()

    first_name = first_candidate_name[0] if first_candidate_name else "Super"
    last_name = (
        second_candidate_name[-1]
        if len(second_candidate_name) > 1
        else second_candidate_name[0]
        if second_candidate_name
        else "Hero"
    )

    middle_name = random.choice(SUPERHERO_MIDDLE_NAMES)
    return f"{first_name} '{middle_name}' {last_name}"


def _generate_superhero_profile(
    candidates_data: List[Dict[str, str]], superhero_name: str
) -> str:
    """Generate superhero profile using LLM.

    Args:
        candidates_data: List of candidate data
        superhero_name: Generated superhero name

    Returns:
        Generated profile text
    """
    llm = OpenAI(
        model=LLM_MODEL,
        temperature=0.3,
        max_tokens=MAX_RESPONSE_TOKENS,
        timeout=LLM_TIMEOUT,
    )

    candidates_info = "\n\n".join(
        [
            f"Candidate {i + 1} ({data['name']}):\n{data['content']}"
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


def _format_superhero_output(
    superhero_name: str, candidates_data: List[Dict[str, str]], profile: str
) -> str:
    """Format the superhero output.

    Args:
        superhero_name: Superhero name
        candidates_data: List of candidate data
        profile: Generated profile

    Returns:
        Formatted output string
    """
    candidate_names = ", ".join([data["name"] for data in candidates_data])

    return f"""
🦸 SUPERHERO CANDIDATE CREATED! 🦸

Name: {superhero_name}
Combined from: {len(candidates_data)} candidates
- {candidate_names}

{"-" * 80}

{profile}

{"-" * 80}

This superhero candidate combines the best qualities from all {len(candidates_data)} candidates!
"""
