from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from IPython.display import Markdown, display
import chromadb

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

# Ensure metadata propagates to nodes
for node in nodes:
    if not node.metadata.get("source_file"):
        # Fallback if metadata didn't propagate
        node.metadata["source_file"] = "unknown"
        node.metadata["doc_type"] = "resume"

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
