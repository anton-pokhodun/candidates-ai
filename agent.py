import json
from typing import Optional
from dotenv import load_dotenv
import chromadb
from chromadb.api import ClientAPI

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
import wikipedia

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
COLLECTION_NAME = "csv"
CHROMA_DB_PATH = "./chroma_db"
LLM_MODEL = "gpt-4o-mini"


# ============================================================================
# Database Client
# ============================================================================
def get_chroma_client() -> ClientAPI:
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


# ============================================================================
# Tool 1: Candidate Retrieval Tool
# ============================================================================
def search_candidates(query: str, top_k: int = 3) -> str:
    """Search for candidates based on a natural language query.

    This tool searches through candidate resumes and returns relevant information
    about candidates matching the query criteria.

    Args:
        query: Natural language search query (e.g., "Python developer with 5 years experience")
        top_k: Number of top results to return (default: 3)

    Returns:
        A formatted string containing candidate information matching the query
    """
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Initialize embedding model
    embed_model = OpenAIEmbedding()

    # Create vector store and index
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    # Perform similarity search
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    if not results:
        return "No candidates found matching your query."

    # Format results with metadata
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


# ============================================================================
# Tool 2: General Knowledge Tool
# ============================================================================
def search_wikipedia(query: str, sentences: int = 3) -> str:
    """Search Wikipedia for general knowledge and factual information.

    This tool searches Wikipedia and returns relevant information about general topics,
    facts, concepts, people, places, and events. Use this for answering general knowledge
    questions unrelated to candidate resumes.

    Args:
        query: Search query or topic (e.g., "Python programming language", "Machine learning")
        sentences: Number of sentences to return from the summary (default: 3)

    Returns:
        A formatted string containing the Wikipedia summary for the topic
    """
    try:
        # Set language to English
        wikipedia.set_lang("en")

        # Search for the topic
        search_results = wikipedia.search(query, results=3)

        if not search_results:
            return f"No Wikipedia articles found for '{query}'."

        # Get summary of the first result
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)

        return f"Wikipedia - {page_title}:\n\n{summary}"

    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation pages
        options = e.options[:5]  # Show first 5 options
        return f"Multiple topics found for '{query}'. Please be more specific. Options include: {', '.join(options)}"

    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'. Please try a different search term."

    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


def search_candidates_structured(query: str, top_k: Optional[int] = 10) -> dict:
    """Search for candidates and return structured data for API."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    embed_model = OpenAIEmbedding()
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    candidates = []
    for result in results:
        candidates.append(
            {
                "candidate_id": str(result.node.metadata.get("candidate_id", "N/A")),
                "candidate_name": result.node.metadata.get("candidate_name", "Unknown"),
                "file_name": result.node.metadata.get("file_name", "unknown"),
                "score": float(result.score),
                "content": result.node.get_content(),
            }
        )

    return {
        "answer": f"Found {len(candidates)} candidates matching your query.",
        "candidates": candidates,
    }


# ============================================================================
# Tool 3: Super Hero Creator Tool
# ============================================================================
def create_superhero(candidate_names: str) -> str:
    """Create a superhero candidate by combining skills from 2-3 candidates.

    This tool takes candidate names and creates a "superhero" candidate
    that combines the best skills and qualifications from all of them.
    The superhero's name consists of the first name of the first candidate,
    a superhero-style middle name, and the last name of the second candidate.

    Args:
        candidate_names: Comma-separated candidate names (e.g., "John Doe,Jane Smith" or "John Doe,Jane Smith,Bob Johnson")

    Returns:
        A formatted string describing the superhero candidate with combined skills
    """
    try:
        # Parse candidate names
        names = [name.strip() for name in candidate_names.split(",")]

        if len(names) < 2 or len(names) > 3:
            return "Error: Please provide 2 or 3 candidate names separated by commas."

        # Initialize ChromaDB
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # Retrieve candidate information
        candidates_data = []
        for candidate_name in names:
            # Query by candidate_name metadata
            results = collection.get(
                where={"candidate_name": candidate_name},
                include=["documents", "metadatas"],
            )

            if not results["documents"]:
                return f"Error: Candidate with name '{candidate_name}' not found."

            # Limit content length per candidate to prevent excessive token usage
            full_content = " ".join(results["documents"])
            max_chars_per_candidate = 3000
            truncated_content = full_content[:max_chars_per_candidate]
            if len(full_content) > max_chars_per_candidate:
                truncated_content += "... [truncated]"

            candidates_data.append(
                {
                    "name": candidate_name,
                    "content": truncated_content,
                }
            )

        # Create superhero name (first name of first + superhero middle name + last name of second)
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

        # Generate a superhero middle name
        import random

        superhero_middle_names = [
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
        middle_name = random.choice(superhero_middle_names)

        superhero_name = f"{first_name} '{middle_name}' {last_name}"

        # Use LLM to extract and combine skills
        llm = OpenAI(
            model=LLM_MODEL,
            temperature=0.3,
            max_tokens=1000,  # Limit response length
            timeout=30.0,  # 30 second timeout
        )

        # Prepare prompt for skill extraction and combination
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

        # Format the final output
        result = f"""
🦸 SUPERHERO CANDIDATE CREATED! 🦸

Name: {superhero_name}
Combined from: {len(candidates_data)} candidates
- {", ".join([data["name"] for data in candidates_data])}

{"-" * 80}

{response.text}

{"-" * 80}

This superhero candidate combines the best qualities from all {len(candidates_data)} candidates!
"""

        return result

    except Exception as e:
        return f"Error creating superhero: {str(e)}"


# ============================================================================
# Agent Setup
# ============================================================================
def create_agent() -> ReActAgent:
    """Create and configure the ReAct agent with all tools."""

    # Initialize LLM
    llm = OpenAI(model=LLM_MODEL, temperature=0.1)
    # Create tools
    tools = [
        FunctionTool.from_defaults(
            fn=search_candidates,
            name="search_candidates",
            description="Search for candidates using natural language queries. Use this to find candidates with specific skills, experience, or qualifications.",
        ),
        FunctionTool.from_defaults(
            fn=search_wikipedia,
            name="search_wikipedia",
            description="Search Wikipedia for general knowledge, facts, and information about topics, concepts, people, places, or events. Use this for questions unrelated to candidate resumes.",
        ),
        FunctionTool.from_defaults(
            fn=create_superhero,
            name="create_superhero",
            description="Create a superhero candidate by combining the best skills from 2-3 candidates. Provide comma-separated candidate IDs (e.g., '123,456' or '123,456,789'). The superhero's name will be the first name of the first candidate and last name of the second candidate.",
        ),
    ]

    # Create agent
    agent = ReActAgent(tools=tools, llm=llm, verbose=True)
    return agent


async def search_with_agent(query: str, top_k: int = 10):
    """Use ReAct agent to search for candidates and return structured results."""
    agent = create_agent()
    ctx = Context(agent)
    print(f"Processing query: {query}")
    # Get structured results first
    structured_results = search_candidates_structured(query, top_k)

    # Send metadata first
    # yield f"data: {json.dumps({'type': 'metadata', 'data': {'candidates': structured_results['candidates']}})}\n\n"

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
