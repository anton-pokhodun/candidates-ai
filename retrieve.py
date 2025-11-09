from llama_index.core import VectorStoreIndex, PromptTemplate
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.openai import OpenAI

load_dotenv()

collection_name = "csv"


def get_chroma_client():
    """Get or create a persistent ChromaDB client."""
    return chromadb.PersistentClient(path="./chroma_db")


def get_query_engine(
    similarity_top_k: int = 3, text_qa_template: PromptTemplate | None = None
):
    """Create a query engine from the persisted ChromaDB collection."""

    # Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
    # Get the persisted collection
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    # Create embedding model (must match the one used during indexing)
    embedding = OpenAIEmbedding()

    # Create vector store from existing collection
    vector_store = ChromaVectorStore(chroma_collection=collection, embedding=embedding)

    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embedding
    )

    # Return query engine
    return index.as_query_engine(
        similarity_top_k=similarity_top_k, text_qa_template=text_qa_template
    )


def get_all_candidates():
    """Retrieve all unique candidate names from the ChromaDB collection."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    # Get all documents from the collection
    all_docs = collection.get(include=["metadatas"])

    # Extract unique candidate names
    candidate_names = set()
    candidate_info = {}

    for metadata in all_docs.get("metadatas", []):
        if metadata and "candidate_name" in metadata:
            name = metadata["candidate_name"]
            candidate_names.add(name)

            # Store additional info for each candidate
            if name not in candidate_info:
                candidate_info[name] = {
                    "name": name,
                    "file_name": metadata.get("file_name", "unknown"),
                    # "profession": metadata.get("profession", "Unknown"),
                }

    return sorted(list(candidate_names)), candidate_info


def get_candidate_by_id(candidate_id: str, use_llm: bool = True):
    """Retrieve detailed information for a specific candidate by ID."""
    chroma_client = get_chroma_client()
    collection = chroma_client.get_collection(name=collection_name)

    # Get all documents from the collection
    all_docs = collection.get(include=["metadatas", "documents"])

    # Find all chunks belonging to this candidate
    candidate_chunks = []
    candidate_metadata = None

    metadatas = all_docs.get("metadatas", [])
    documents = all_docs.get("documents", [])

    for i, metadata in enumerate(metadatas):
        if metadata and metadata.get("candidate_name") == candidate_id:
            if candidate_metadata is None:
                candidate_metadata = {
                    "name": metadata.get("candidate_name"),
                    "file_name": metadata.get("file_name", "unknown"),
                    "profession": metadata.get("profession", "Unknown"),
                }

            if i < len(documents):
                candidate_chunks.append(documents[i])

    if candidate_metadata is None:
        return None

    # Combine all chunks to get full CV text
    full_text = "\n\n".join(candidate_chunks)

    result = {
        **candidate_metadata,
        "chunks_count": len(candidate_chunks),
        "full_text": full_text,
        "chunks": candidate_chunks,
    }

    # Use LLM to generate a formatted summary
    if use_llm and full_text:
        llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

        prompt = f"""Based on the following CV information, create a well-structured professional summary.

CV Content:
{full_text}

Please provide a comprehensive summary with the following sections:
1. **Full Name**: Extract the candidate's full name
2. **Current Position**: Current or most recent job title and company
3. **Professional Summary**: A brief 2-3 sentence overview of their career
4. **Years of Experience**: Calculate total years of professional experience based on employment dates
5. **Key Skills**: List all technical skills, tools, frameworks, and technologies (organized by category if applicable)
6. **Work Experience**: Summarize each position with company name, role, dates, and key responsibilities/achievements
7. **Education**: Degrees, institutions, and graduation years
8. **Certifications**: Any professional certifications or additional training
9. **Notable Achievements**: Key accomplishments or projects worth highlighting

Format the response in a clear, professional manner using markdown. Be concise but thorough.
If any information is not available in the CV, indicate "Not specified" for that section.
"""

        llm_response = llm.complete(prompt)
        result["formatted_summary"] = str(llm_response)

    return result


def print_retrieved_chunks(response):
    """Display retrieved chunks with scores and metadata."""
    print("\n" + "=" * 80)
    print(f"RETRIEVED CHUNKS ({len(response.source_nodes)} total)")
    print("=" * 80)

    for i, node in enumerate(response.source_nodes, 1):
        print(f"\n--- Chunk {i} (Score: {node.score:.4f}) ---")
        print(f"Text: {node.text}")
        if node.metadata:
            print(f"Metadata: {node.metadata}")


if __name__ == "__main__":
    list_prompt_tmpl = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "CRITICAL: You MUST extract ALL unique candidates from the context above. DO NOT skip any candidates.\n\n"
        "For each candidate, provide:\n"
        "- Source File: Extract the source_file or file_path from the chunk metadata\n"
        "- Full name (if not available in the CV, assign a unique placeholder name from Hollywood stars: use names like 'Brad Pitt', 'Tom Cruise', 'Leonardo DiCaprio', etc. - make sure each candidate gets a DIFFERENT name)\n"
        "- Current profession or job title (use ACTUAL information from CV)\n"
        "- Years of commercial experience: Calculate total years based on ACTUAL CV information:\n"
        "  * If CV explicitly states total experience (e.g., '5 years of experience', '3+ years'), use that number\n"
        "  * If CV lists employment periods with dates, calculate the difference between earliest start date and latest end date\n"
        "  * Accept date formats: 'Month Year' (June 2015), 'Year/Month' (2015/06), 'MM/YYYY', 'YYYY-MM', etc.\n"
        "  * Example: 'June 2014 - August 2015' + 'March 2016 - Present' = calculate from June 2014 to Present\n"
        "  * If only years are given (e.g., '2014-2016', '2017-2020'), sum the periods\n"
        "  * Round to nearest whole number or use decimals (e.g., 2.5 years)\n"
        "  * If no experience information is found, state 'Not specified'\n"
        "- Key skills: Extract ALL skills mentioned in the CV, including:\n"
        "  * Programming languages (e.g., Python, Java, JavaScript, C++)\n"
        "  * Frameworks and libraries (e.g., React, Django, Spring, TensorFlow)\n"
        "  * Tools and technologies (e.g., Docker, Kubernetes, Git, AWS)\n"
        "  * Databases (e.g., PostgreSQL, MongoDB, MySQL)\n"
        "  * Methodologies (e.g., Agile, Scrum, TDD)\n"
        "  * Soft skills (e.g., Leadership, Communication, Team collaboration)\n"
        "  * Domain knowledge (e.g., Machine Learning, Cloud Architecture, DevOps)\n"
        "  * Certifications and specializations\n"
        "  * List all skills found, separated by commas\n"
        "  * If no skills are found, state 'Not specified'\n"
        "- Education (use ACTUAL information from CV, if mentioned)\n\n"
        "IMPORTANT: \n"
        "1. Use real CV data for all fields except the name. Only replace missing names with Hollywood star names.\n"
        "2. You MUST list EVERY SINGLE candidate - if you see 20 different source files, list ALL 20 candidates.\n"
        "3. Format the response as a structured list with these fields clearly labeled.\n"
        "4. Each candidate should be separated by a blank line.\n"
        "5. Start your response with: 'Total candidates found: [NUMBER]'\n\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    query_engine = get_query_engine(
        similarity_top_k=10, text_qa_template=list_prompt_tmpl
    )
    query = "I want to find a teacher of history. If there are any - provide me a summary of their background."

    response = query_engine.query(query)
    print(response)
