# Candidates AI
A RAG (Retrieval-Augmented Generation) application for intelligent candidate CV analysis and summarization using vector embeddings and LLM-powered insights.

## Overview

Candidates AI allows you to:
- Ingest and process multiple CV documents
- Store candidate information with vector embeddings for semantic search
- Generate AI-powered summaries of candidate profiles
- Anonymize candidate data using pseudonyms

## Technologies Used

### Backend
- **FastAPI** - Modern Python web framework for building APIs
- **LlamaIndex** - Framework for LLM-powered data applications
- **ChromaDB** - Vector database for embeddings storage
- **OpenAI** - Embeddings (text-embedding-ada-002) and LLM (GPT-4o-mini)
- **Python 3.x** - Core programming language

### Frontend
- **Vanilla JavaScript** - Simple, dependency-free frontend
- **Server-Sent Events (SSE)** - Real-time streaming of AI responses
- **Python HTTP Server** - Static file serving

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- pip (Python package manager)

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd candidates-ai
```

2. **Run the setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   The setup script will:
   - Create a `.env` file (you'll need to add your OpenAI API key)
   - Install Python dependencies
   - Create necessary directories (`chroma_db`, `data`)
   - Ingest CV documents from the `data` directory

3. **Configure OpenAI API Key**

   Update the `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

## Running the Application

### Start the Backend Server

```bash
make run
```

Or manually:
```bash
fastapi dev backend.py
```


### Start the Frontend Server

In a separate terminal:
```bash
python -m http.server 8080
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:8080
```

## Project Structure

```
candidates-ai/
├── persist.py          # Data ingestion and vector store creation
├── service.py          # Core business logic and LLM integration
├── backend.py          # FastAPI application and routes
├── setup.sh            # Automated setup script
├── requirements.txt    # Python dependencies
├── data/               # Directory for CV documents (PDFs, text files, etc.)
├── chroma_db/          # ChromaDB vector store (created automatically)
└── .env                # Environment variables (create from setup)
```

## Usage

1. Place CV documents (PDF, TXT, etc.) in the `data/` directory
2. Run the setup script to ingest and process the documents
3. Start both backend and frontend servers
4. Browse candidates and generate AI-powered summaries through the web interface

---

Built with ❤️ using LlamaIndex and OpenAI

