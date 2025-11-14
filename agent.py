"""ReAct agent for candidate search with integrated tools."""

import json
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import ReActAgent, AgentStream
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

from config import LLM_MODEL
from tools import (
    search_candidates,
    search_wikipedia,
    create_superhero,
    search_candidates_structured,
)

load_dotenv()


def create_agent() -> ReActAgent:
    """Create and configure the ReAct agent with all tools.

    Returns:
        Configured ReActAgent instance
    """
    llm = OpenAI(model=LLM_MODEL, temperature=0.1)

    tools = [
        FunctionTool.from_defaults(
            fn=search_candidates,
            name="search_candidates",
            description=(
                "Search for candidates using natural language queries. "
                "Use this to find candidates with specific skills, experience, or qualifications."
            ),
        ),
        FunctionTool.from_defaults(
            fn=search_wikipedia,
            name="search_wikipedia",
            description=(
                "Search Wikipedia for general knowledge, facts, and information about topics, "
                "concepts, people, places, or events. Use this for questions unrelated to candidate resumes."
            ),
        ),
        FunctionTool.from_defaults(
            fn=create_superhero,
            name="create_superhero",
            description=(
                "Create a superhero candidate by combining the best skills from 2-3 candidates. "
                "Provide comma-separated candidate names (e.g., 'John Doe,Jane Smith'). "
                "The superhero's name will be the first name of the first candidate "
                "and last name of the second candidate."
            ),
        ),
    ]

    return ReActAgent(tools=tools, llm=llm, verbose=True)


async def search_with_agent(query: str, top_k: int = 10):
    """Use ReAct agent to search for candidates and return structured results.

    Args:
        query: Search query
        top_k: Number of top results to return

    Yields:
        Server-sent events with streaming response
    """
    agent = create_agent()
    ctx = Context(agent)

    print(f"Processing query: {query}")

    # Get structured results (not currently used but available)
    # structured_results = search_candidates_structured(query, top_k)

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
