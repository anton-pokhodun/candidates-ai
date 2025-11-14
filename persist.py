"""Script to build and persist the vector index from documents."""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode
from dotenv import load_dotenv
from typing import Dict, List
import chromadb
import random

from db_utils import get_chroma_client, get_embedding_model, reset_collection
from config import COLLECTION_NAME, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()


# ============================================================================
# Famous Names for Anonymization
# ============================================================================
FAMOUS_NAMES = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "Isaac Newton",
    "Nikola Tesla",
    "Ada Lovelace",
    "Alan Turing",
    "Grace Hopper",
    "Stephen Hawking",
    "Carl Sagan",
    "Neil Armstrong",
    "Sally Ride",
    "Rosa Parks",
    "Martin Luther King Jr.",
    "Nelson Mandela",
    "Mahatma Gandhi",
    "Mother Teresa",
    "Malala Yousafzai",
    "Winston Churchill",
    "Abraham Lincoln",
    "George Washington",
    "Thomas Edison",
    "Alexander Graham Bell",
    "Wright Brothers",
    "Henry Ford",
    "Steve Jobs",
    "Bill Gates",
    "Elon Musk",
    "Mark Zuckerberg",
    "Jeff Bezos",
    "Oprah Winfrey",
    "Walt Disney",
    "Pablo Picasso",
    "Vincent van Gogh",
    "Frida Kahlo",
    "Claude Monet",
    "Wolfgang Mozart",
    "Ludwig van Beethoven",
    "Johann Bach",
    "Elvis Presley",
    "Michael Jackson",
    "The Beatles",
    "Bob Dylan",
    "Freddie Mercury",
    "Muhammad Ali",
    "Serena Williams",
    "Lionel Messi",
    "Michael Jordan",
    "Bruce Lee",
    "Jane Austen",
    "William Shakespeare",
    "Charles Dickens",
    "Mark Twain",
    "Ernest Hemingway",
    "Maya Angelou",
    "J.K. Rowling",
    "Charles Darwin",
    "Galileo Galilei",
    "Copernicus",
    "Johannes Kepler",
    "Benjamin Franklin",
    "Eleanor Roosevelt",
    "Cleopatra",
    "Julius Caesar",
    "Alexander the Great",
    "Napoleon Bonaparte",
    "Queen Elizabeth I",
    "Catherine the Great",
    "Confucius",
    "Buddha",
    "Socrates",
    "Plato",
    "Aristotle",
    "Pythagoras",
]


# ============================================================================
# Document Loading
# ============================================================================
def load_documents(data_dir: str) -> List[Document]:
    """Load documents from the specified directory.

    Args:
        data_dir: Path to directory containing documents

    Returns:
        List of Document objects
    """
    print(f"Loading documents from {data_dir}...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"Loaded {len(documents)} documents")
    return documents


# ============================================================================
# Document Chunking
# ============================================================================
def create_chunks(documents: List[Document]) -> List[BaseNode]:
    """Parse documents into chunks/nodes.

    Args:
        documents: List of documents to chunk

    Returns:
        List of TextNode chunks
    """
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" ",
        paragraph_separator="\n\n",
    )

    print("Creating chunks...")
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} chunks")
    return nodes


# ============================================================================
# Metadata Assignment
# ============================================================================
def assign_candidate_metadata(documents: List[Document], nodes: List[BaseNode]) -> None:
    """Assign anonymized candidate names and IDs to documents and nodes.

    Modifies nodes in-place to add candidate_name and candidate_id metadata.

    Args:
        documents: List of original documents
        nodes: List of chunks/nodes to add metadata to
    """
    # Shuffle and assign unique famous names to documents
    random.shuffle(FAMOUS_NAMES)
    candidate_ids: Dict[str, int] = {}
    candidate_names: Dict[str, str] = {}

    for idx, doc in enumerate(documents):
        file_name = doc.metadata.get("file_name", "unknown")

        candidate_ids[doc.doc_id] = random.randint(1000, 9999)
        # Assign from shuffled famous names list
        assigned_name = FAMOUS_NAMES[idx % len(FAMOUS_NAMES)]
        candidate_names[doc.doc_id] = assigned_name
        print(f"Assigned name '{assigned_name}' to {file_name}")

    # Add metadata to each node
    for node in nodes:
        if not node.metadata.get("file_name"):
            node.metadata["file_name"] = "unknown"

        ref_doc_id = node.ref_doc_id
        if ref_doc_id is not None:
            node.metadata["candidate_name"] = candidate_names.get(ref_doc_id, "Unknown")
            node.metadata["candidate_id"] = candidate_ids.get(ref_doc_id, 0)
        else:
            node.metadata["candidate_name"] = "Unknown"
            node.metadata["candidate_id"] = 0

    if nodes:
        print(f"\nSample chunk metadata: {nodes[0].metadata}")
        print(f"Sample chunk text (first 200 chars): {nodes[0].get_content()[:200]}...")


# ============================================================================
# Index Creation
# ============================================================================
def create_and_persist_index(
    nodes: List[BaseNode], collection: chromadb.Collection
) -> VectorStoreIndex:
    """Create vector store index and persist to ChromaDB.

    Args:
        nodes: List of chunks/nodes to index
        collection: ChromaDB collection to store vectors

    Returns:
        VectorStoreIndex: Created index
    """
    embed_model = get_embedding_model()
    vector_store = ChromaVectorStore(
        chroma_collection=collection, embedding=embed_model
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Building index and persisting to ChromaDB...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print("Index created and persisted successfully!")
    return index


# ============================================================================
# Main Execution
# ============================================================================
def main() -> None:
    """Main execution function to build and persist the vector index."""
    # Initialize ChromaDB client and reset collection
    chroma_client = get_chroma_client()
    collection = reset_collection(chroma_client, COLLECTION_NAME)

    # Load and process documents
    documents = load_documents(DATA_DIR)
    nodes = create_chunks(documents)
    assign_candidate_metadata(documents, nodes)

    # Create and persist index
    index = create_and_persist_index(nodes, collection)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Index Creation Summary")
    print(f"{'=' * 60}")
    print(f"Total documents indexed: {len(documents)}")
    print(f"Total chunks created: {len(nodes)}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Data directory: {DATA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
