from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


CATALOG_PATH = Path(__file__).parent / "catalog" / "nodes.yaml"


class ArtifactSpec(BaseModel):
    name: str
    path: Optional[str] = None


class Artifacts(BaseModel):
    inputs: List[ArtifactSpec] = Field(default_factory=list)
    outputs: List[ArtifactSpec] = Field(default_factory=list)


class NodeTemplate(BaseModel):
    name: str
    type: str
    parameters: List[str] = Field(default_factory=list)
    artifacts: Artifacts
    limits: Optional[Dict] = None
    version: Optional[str] = None


def load_catalog() -> List[NodeTemplate]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found at {CATALOG_PATH}")
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to read catalog: {e}")

    nodes = raw.get("nodes", [])
    try:
        return [NodeTemplate(**node) for node in nodes]
    except Exception as e:
        raise RuntimeError(f"Invalid catalog format: {e}")


app = FastAPI(title="Synthetic Data Suite Backend", version="0.1.0")

# Allow CORS in dev so the frontend can call the API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/workflow-templates", response_model=List[NodeTemplate])
def get_workflow_templates() -> List[NodeTemplate]:
    try:
        return load_catalog()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
