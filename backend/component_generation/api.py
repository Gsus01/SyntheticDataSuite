from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from component_generation.ingest import (
    DEFAULT_MAX_CHARS,
    ingest_paths,
)


router = APIRouter(prefix="/component-generation", tags=["component-generation"])


class IngestFileInfo(BaseModel):
    filename: str
    size: int
    truncated: bool = False
    content_type: str | None = Field(default=None, alias="contentType")


class IngestResponse(BaseModel):
    combined: str
    files: List[IngestFileInfo]


def _safe_filename(name: str | None, *, fallback: str) -> str:
    if not name:
        return fallback
    return Path(name).name or fallback


@router.post("/ingest", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    include_markdown: bool = Query(False, alias="includeMarkdown"),
    max_chars_per_file: int = Query(DEFAULT_MAX_CHARS, alias="maxCharsPerFile", gt=0),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    file_infos: List[IngestFileInfo] = []
    temp_paths: List[Path] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for idx, upload in enumerate(files, start=1):
            filename = _safe_filename(upload.filename, fallback=f"input-{idx}")
            path = temp_root / filename
            try:
                data = await upload.read()
            finally:
                await upload.close()
            path.write_bytes(data)
            temp_paths.append(path)
            file_infos.append(
                IngestFileInfo(
                    filename=filename,
                    size=len(data),
                    contentType=upload.content_type,
                )
            )

        try:
            combined = ingest_paths(
                temp_paths,
                max_chars_per_file=max_chars_per_file,
                include_markdown=include_markdown,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(combined=combined, files=file_infos)
