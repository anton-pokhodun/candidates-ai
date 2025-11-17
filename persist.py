"""Simplified and verified index builder."""

from typing import List, Dict, Tuple
import random
from dotenv import load_dotenv
import chromadb
import re

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode
from llama_index.core.llms import ChatMessage

from db_utils import get_chroma_client, get_embedding_model, reset_collection, get_llm
from config import COLLECTION_NAME, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, FAMOUS_NAMES

load_dotenv()


def extract_profession(text: str, llm_fallback=True) -> str:
    """
    Extract the candidate's profession from a CV using a combined approach:
    1. Rule-based extraction from common headers/keywords.
    2. Optional LLM fallback if profession cannot be determined.
    """

    # ------------------------------
    # 1️⃣ Rule-based extraction
    # ------------------------------
    patterns = [
        r"(?i)current position[:\-]\s*(.*)",
        r"(?i)profession[:\-]\s*(.*)",
        r"(?i)job title[:\-]\s*(.*)",
        r"(?i)role[:\-]\s*(.*)",
        r"(?i)^title[:\-]\s*(.*)",  # line starts with Title
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            prof = match.group(1).strip()
            # simple validation
            if prof and len(prof) <= 100:
                print(f"Extracted profession using rule-based method: {prof}")
                return prof

    # Optional: look for common CV starting lines (heuristic)
    first_line = text.strip().split("\n")[0]
    if 2 <= len(first_line.split()) <= 6:  # likely a title line
        return first_line.strip()[:100]

    # ------------------------------
    # 2️⃣ LLM fallback
    # ------------------------------
    if llm_fallback:
        try:
            llm = get_llm()
            cv_excerpt = text[:2000]  # focus on first part of CV
            prompt = f"""
Extract the candidate's current profession or job title from this CV excerpt.
Return ONLY the job title/profession, nothing else. If unclear, return "Not Specified".

CV excerpt:
{cv_excerpt}

Profession:
"""
            messages = [ChatMessage(role="user", content=prompt)]
            response = llm.chat(messages)
            profession = response.message.content.strip()
            if profession and len(profession) <= 100:
                return profession
        except Exception as e:
            print(f"LLM fallback failed: {e}")

    # ------------------------------
    # Default if nothing found
    # ------------------------------
    return "Not Specified"


# =====================================================================
# Load documents
# =====================================================================
def load_documents(data_dir: str) -> List[Document]:
    print(f"Loading documents from {data_dir}...")
    return SimpleDirectoryReader(data_dir).load_data()


# =====================================================================
# Chunk documents
# =====================================================================
def create_chunks(documents: List[Document]) -> List[BaseNode]:
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" ",
        paragraph_separator="\n\n",
    )
    print("Chunking documents...")
    return splitter.get_nodes_from_documents(documents)


# =====================================================================
# Assign metadata (critical fix)
# =====================================================================
def assign_candidate_metadata(documents: List[Document], nodes: List[BaseNode]) -> None:
    """
    Ensures:
    - Every file becomes exactly 1 candidate.
    - ALL nodes referencing that file get metadata: candidate_id, candidate_name, profession.
    - No missing metadata ever.
    """

    print("Assigning candidate metadata...")

    famous = FAMOUS_NAMES.copy()
    random.shuffle(famous)

    # Map file paths → candidate info
    file_to_candidate: Dict[str, Tuple[str, int, str]] = {}

    unique_files = list(
        {doc.metadata.get("file_path", doc.doc_id) for doc in documents}
    )

    if len(unique_files) > len(famous):
        repeat = (len(unique_files) // len(famous)) + 1
        famous = famous * repeat

    # Assign one candidate per unique file
    for i, doc in enumerate(documents):
        file_path = doc.metadata.get("file_path", doc.doc_id)

        if file_path not in file_to_candidate:
            name = famous[len(file_to_candidate)]
            cid = random.randint(1000, 9999)
            profession = extract_profession(doc.get_content())

            file_to_candidate[file_path] = (name, cid, profession)

    # Apply metadata to EVERY chunk
    for node in nodes:
        file_path = node.metadata.get("file_path") or node.ref_doc_id
        name, cid, profession = file_to_candidate[file_path]

        node.metadata["candidate_name"] = name
        node.metadata["candidate_id"] = cid
        node.metadata["profession"] = profession.lower()
        node.metadata.setdefault("file_name", file_path)
        node.set_content(node.get_content().strip())

    print(f"Example chunk metadata:\n{nodes[0].metadata}")


# =====================================================================
# Build & persist index
# =====================================================================
def create_and_persist_index(nodes: List[BaseNode], collection: chromadb.Collection):
    embed_model = get_embedding_model()
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )
    ctx = StorageContext.from_defaults(vector_store=vector_store)

    print("Creating vector index...")
    index = VectorStoreIndex(nodes=nodes, storage_context=ctx, embed_model=embed_model)
    print("Index successfully saved.")
    return index


# =====================================================================
# MAIN
# =====================================================================
def main():
    client = get_chroma_client()
    collection = reset_collection(client, COLLECTION_NAME)

    docs = load_documents(DATA_DIR)
    nodes = create_chunks(docs)
    assign_candidate_metadata(docs, nodes)
    index = create_and_persist_index(nodes, collection)

    print("\n===== Index Summary =====")
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(nodes)}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
