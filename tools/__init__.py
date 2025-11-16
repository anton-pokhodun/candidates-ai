"""Tools package for the candidate search agent."""

from .candidate_search import search_candidates
from .wikipedia_search import search_wikipedia
from .superhero_creator import create_superhero

__all__ = [
    "search_candidates",
    "search_wikipedia",
    "create_superhero",
]
