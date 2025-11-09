from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
import chromadb
import re

load_dotenv()

_chroma_client = None
collection_name = "csv"


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return _chroma_client


# load some documents
chroma_client = get_chroma_client()
try:
    chroma_client.delete_collection(name=collection_name)
    print(f"Deleted existing collection '{collection_name}'")
except Exception:
    pass


collection = chroma_client.get_or_create_collection(name=collection_name)

# Load documents from data folder
print("Loading documents...")
documents = SimpleDirectoryReader("./data").load_data()
print(f"Loaded {len(documents)} documents")

# Chunk size: 512-1024 tokens works well for CV sections (experience, skills, education)
# Larger overlap ensures continuity between chunks
node_parser = SentenceSplitter(
    chunk_size=1024,  # Reduced from 2024 for better semantic coherence
    chunk_overlap=200,
    separator=" ",
    paragraph_separator="\n\n",
)

# Parse documents into chunks/nodes
print("Creating chunks...")
nodes = node_parser.get_nodes_from_documents(documents)
print(f"Created {len(nodes)} chunks")


def extract_candidate_name(text, file_name):
    """Extract candidate name from CV text or use file name."""
    # Try to find name patterns in the first 500 characters
    text_start = text[:500].strip()

    name_pattern = (
        r"(?:Name|Full Name|Candidate Name)[\s:]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"
    )
    match = re.search(name_pattern, text_start, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    if file_name and file_name != "unknown":
        name_from_file = (
            file_name.replace(".pdf", "").replace(".txt", "").replace(".docx", "")
        )
        name_from_file = name_from_file.replace("_", " ").replace("-", " ").strip()
        # If it looks like a name, use it
        if re.match(r"^[A-Za-z\s]+$", name_from_file) and len(name_from_file) < 50:
            return name_from_file.title()

    return None


# Extract candidate names from documents
candidate_names = {}
candidate_counter = 1

for doc in documents:
    file_name = doc.metadata.get("file_name", "unknown")
    extracted_name = extract_candidate_name(doc.text, file_name)

    if extracted_name:
        candidate_names[doc.doc_id] = extracted_name
        print(f"Found candidate name: {extracted_name} from {file_name}")
    else:
        candidate_names[doc.doc_id] = f"candidate_{candidate_counter}"
        print(f"No name found in {file_name}, using candidate_{candidate_counter}")
        candidate_counter += 1


for node in nodes:
    if not node.metadata.get("file_name"):
        node.metadata["file_name"] = "unknown"

    ref_doc_id = node.ref_doc_id
    node.metadata["candidate_name"] = candidate_names.get(ref_doc_id, "Candidate#0")

    if not node.metadata.get("profession"):
        node.metadata["profession"] = "Unknown Profession"

if nodes:
    print(f"\nSample chunk metadata: {nodes[0].metadata}")
    print(f"Sample chunk text (first 200 chars): {nodes[0].text[:200]}...")


# Create vector store and storage context
embedding = OpenAIEmbedding()
vector_store = ChromaVectorStore(chroma_collection=collection, embedding=embedding)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index and persist to ChromaDB
print("Building index and persisting to ChromaDB...")
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embedding,
    show_progress=True,
)
print("Index created and persisted successfully!")
print(f"Total documents indexed: {len(documents)}")
print(f"Total chunks created: {len(nodes)}")
