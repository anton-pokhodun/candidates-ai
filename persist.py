from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
import chromadb
import re
import random

load_dotenv()

_chroma_client = None
collection_name = "csv"

# Famous people names list
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


# Shuffle and assign unique famous names to documents
random.shuffle(FAMOUS_NAMES)
candidate_ids = {}
candidate_names = {}

for idx, doc in enumerate(documents):
    file_name = doc.metadata.get("file_name", "unknown")

    candidate_ids[doc.doc_id] = random.randint(1000, 9999)
    # Assign from shuffled famous names list
    assigned_name = FAMOUS_NAMES[idx % len(FAMOUS_NAMES)]
    candidate_names[doc.doc_id] = assigned_name
    print(f"Assigned name '{assigned_name}' to {file_name}")


# Extract candidate names from documents
candidate_counter = 1
for node in nodes:
    if not node.metadata.get("file_name"):
        node.metadata["file_name"] = "unknown"

    ref_doc_id = node.ref_doc_id
    node.metadata["candidate_name"] = candidate_names.get(ref_doc_id, "Joker :D")
    node.metadata["candidate_id"] = candidate_ids.get(ref_doc_id, 0)


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
