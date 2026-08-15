import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import config
from ingest import get_vector_store, process_and_index_pdfs
from rag import answer_financial_query

app = FastAPI(title="Financial RAG API", version="1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20)


@app.post("/ingest")
async def ingest_files(files: list[UploadFile] = File(...)):
    if len(files) > config.MAX_FILES_PER_BATCH:
        raise HTTPException(400, f"Max {config.MAX_FILES_PER_BATCH} files per batch.")

    temp_paths = []
    try:
        for file in files:
            content = await file.read()
            if len(content) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(400, f"{file.filename} exceeds {config.MAX_FILE_SIZE_MB}MB limit.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                temp_paths.append(tmp.name)

        result = process_and_index_pdfs(temp_paths)
        return {
            "files_indexed": result.files_indexed,
            "chunks_indexed": result.chunks_indexed,
            "files_skipped_duplicate": result.files_skipped_duplicate,
            "files_skipped_empty": result.files_skipped_empty,
            "errors": result.errors,
        }
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)


@app.post("/ask")
async def ask_question(request: QueryRequest):
    answer, sources = answer_financial_query(request.question, top_k=request.top_k)
    return {"answer": answer, "sources": sources}


@app.get("/stats")
async def get_stats():
    vs = get_vector_store()
    total_chunks = vs._collection.count()
    return {
        "collection_name": vs._collection.name,
        "total_chunks": total_chunks,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
    }
