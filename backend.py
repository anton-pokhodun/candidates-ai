from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from retrieve import get_all_candidates, get_candidate_by_id

app = FastAPI(title="Candidates AI API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandidateInfo(BaseModel):
    name: str
    file_name: str
    profession: str


class CandidateDetail(BaseModel):
    name: str
    file_name: str
    profession: str
    # chunks_count: int
    # full_text: str
    # chunks: List[str]
    formatted_summary: Optional[str] = None


class CandidatesResponse(BaseModel):
    total: int
    candidates: List[CandidateInfo]


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/candidates", response_model=CandidatesResponse)
async def get_candidates():
    try:
        all, data = get_all_candidates()
        candidates = [
            CandidateInfo(
                name=name,
                file_name=info.get("file_name", "unknown"),
                profession=info.get("profession", "Unknown"),
            )
            for name, info in data.items()
        ]

        return CandidatesResponse(
            total=len(candidates), candidates=sorted(candidates, key=lambda x: x.name)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving candidates: {str(e)}"
        )


@app.get("/candidates/{candidate_id}", response_model=CandidateDetail)
async def get_candidate_details(candidate_id: str, use_llm: bool = True):
    """
    Retrieve detailed information for a specific candidate by ID.

    Args:
        candidate_id: The candidate's ID (e.g., "Candidate#1" or actual name)
        use_llm: Whether to generate formatted summary using LLM (default: True)
    """
    try:
        candidate_data = get_candidate_by_id(candidate_id, use_llm=use_llm)

        if candidate_data is None:
            raise HTTPException(
                status_code=404, detail=f"Candidate '{candidate_id}' not found"
            )

        return CandidateDetail(**candidate_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving candidate details: {str(e)}"
        )
