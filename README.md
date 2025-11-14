# Candidates AI

An AI-powered candidate management system that allows you to search, filter, and analyze candidate CVs using natural language queries and intelligent agents.

https://github.com/user-attachments/assets/5113451b-cc40-4894-964d-ef4cbda0700c




https://github.com/user-attachments/assets/71a6e150-f017-4f1d-9bab-5c42a42a887d

## Features

- **Candidate Management**: Browse and search through candidate profiles
- **AI-Powered Search**: Use natural language to find candidates based on skills, experience, and qualifications
- **Intelligent Agent Tools**: Multi-purpose agent with Wikipedia integration and creative candidate analysis
- **Streaming Responses**: Real-time streaming of AI-generated summaries and search results
- **Vector Search**: Semantic search capabilities using ChromaDB and OpenAI embeddings

## Architecture

### Backend
- **FastAPI**: RESTful API server with streaming support
- **ChromaDB**: Vector database for storing and querying candidate embeddings
- **LlamaIndex**: Framework for building LLM applications with RAG (Retrieval-Augmented Generation)
- **OpenAI**: GPT-4o-mini for language processing and embeddings

### Frontend
- **Vanilla JavaScript**: Lightweight single-page application
- **Server-Sent Events (SSE)**: Real-time streaming of AI responses
- **Responsive Design**: Clean, modern UI for candidate browsing and search

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Backend Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd candidates-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

5. Prepare your candidate data:
   - Place PDF CVs in the `CVs/` directory
   - Run the ingestion script to populate the database:
```bash
python persist.py
```

6. Start the backend server:
```bash
fastapi dev backend.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Serve the frontend (using Python's built-in server):
```bash
python3 -m http.server 8080       
```

## Agent Tools

The AI search agent comes equipped with several specialized tools:

### 1. Semantic Search Tool
Performs vector similarity search across candidate CVs to find relevant matches based on semantic meaning.

**Use cases:**
- "Find candidates with Python and machine learning experience"
- "Search for senior frontend developers"
- "Who has experience with AWS and Docker?"

### 2. Wikipedia Search Tool
Integrates with Wikipedia to enrich queries with external knowledge.

**Use cases:**
- "Search Wikipedia for React framework and find candidates who know it"
- "What is DevOps and which candidates have this experience?"
- "Find information about Kubernetes and matching candidates"

### 3. Superhero Generator Tool
Creates creative superhero personas based on candidate skills and experience.

**Use cases:**
- "Create a superhero based on candidate #5's skills"
- "Generate a superhero for the candidate with the most Python experience"
- "Make a superhero profile for John Doe"

The agent intelligently selects and chains these tools based on your natural language query, providing comprehensive and contextual answers.

## API Endpoints

### GET /candidates
Returns a list of all candidates in the database.

**Response:**
```json
{
  "total": 10,
  "candidates": [
    {
      "candidate_id": 1,
      "candidate_name": "John Doe",
      "file_name": "john_doe.pdf",
      "profession": "Software Engineer"
    }
  ]
}
```

### GET /candidates/{candidate_id}
Streams detailed candidate information with AI-generated summary (SSE).

**Stream Events:**
- `metadata`: Candidate basic information
- `content`: Streaming summary chunks
- `done`: Stream completion

### POST /search
Performs AI-powered semantic search across candidates (SSE).

**Request:**
```json
{
  "query": "Find Python developers with 5+ years experience",
  "top_k": 10
}
```

**Stream Events:**
- `metadata`: Matching candidates with relevance scores
- `content`: Streaming AI-generated answer
- `done`: Stream completion

## Configuration

Key configuration options in `config.py`:

- `CHROMA_DB_PATH`: Path to ChromaDB storage
- `COLLECTION_NAME`: Name of the vector collection
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-small)
- `LLM_MODEL`: Language model for generation (default: gpt-4o-mini)
- `CHUNK_SIZE`: Document chunking size for embeddings
- `CHUNK_OVERLAP`: Overlap between chunks

## Project Structure

```
candidates-ai/
├── backend.py           # FastAPI server and endpoints
├── service.py           # Business logic for candidate operations
├── agent.py             # AI agent with tool definitions
├── db_utils.py          # ChromaDB utilities
├── config.py            # Configuration settings
├── ingest.py            # CV ingestion pipeline
├── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html       # Main HTML file
│   ├── index.js         # JavaScript application
│   └── styles.css       # Styling
├── CVs/                 # PDF candidate files
└── chroma_db/           # ChromaDB storage (generated)
```

Built with ❤️ using LlamaIndex and OpenAI

