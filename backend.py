from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from service import get_all_candidates, generate_candidate_summary_stream
from agent import search_with_agent
from fastapi.responses import StreamingResponse

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
    candidate_id: int
    candidate_name: str
    file_name: str


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


class SearchResult(BaseModel):
    candidate_id: str
    candidate_name: str
    file_name: str
    score: float
    content: str


class QuerySearchResponse(BaseModel):
    answer: str
    matching_candidates: List[SearchResult]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/candidates", response_model=CandidatesResponse)
async def get_candidates():
    try:
        data = get_all_candidates()
        candidates = [
            CandidateInfo(
                candidate_id=id,
                candidate_name=info.get("candidate_name", "unknown"),
                file_name=info.get("file_name", "unknown"),
            )
            for id, info in data.items()
        ]

        return CandidatesResponse(
            total=len(candidates),
            candidates=sorted(candidates, key=lambda x: x.candidate_name),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving candidates: {str(e)}"
        )


@app.get("/candidates/{candidate_id}", response_model=CandidateDetail)
async def get_candidate_details(candidate_id: str, use_llm: bool = True):
    """
    Stream candidate details with LLM-generated summary.
    Returns Server-Sent Events (SSE) stream.
    """
    try:
        return StreamingResponse(
            generate_candidate_summary_stream(candidate_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error streaming candidate details: {str(e)}"
        )


@app.post("/search", response_model=QuerySearchResponse)
async def search_candidates_endpoint(request: QueryRequest):
    """
    Search for candidates using natural language query.
    """
    try:
        results = await search_with_agent(request.query)

        return QuerySearchResponse(
            answer=results["answer"],
            matching_candidates=[
                SearchResult(
                    candidate_id=r["candidate_id"],
                    candidate_name=r["candidate_name"],
                    file_name=r["file_name"],
                    score=r["score"],
                    content=r["content"],
                )
                for r in results["candidates"]
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching candidates: {str(e)}"
        )
