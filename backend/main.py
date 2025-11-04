import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


CATALOG_PATH = Path(__file__).parent / "catalog" / "nodes.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


logger = logging.getLogger(__name__)


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
    parameter_defaults: Optional[Dict[str, Any]] = None


def load_catalog() -> List[NodeTemplate]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog file not found at {CATALOG_PATH}")
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to read catalog: {e}")

    nodes = raw.get("nodes", [])
    templates: List[NodeTemplate] = []

    for node in nodes:
        node_data = dict(node)
        param_defaults = None
        params_file = node_data.pop("parameters_file", None)

        if params_file:
            params_path = Path(params_file)
            if not params_path.is_absolute():
                params_path = PROJECT_ROOT / params_path
            try:
                with params_path.open("r", encoding="utf-8") as pf:
                    param_defaults = json.load(pf)
            except FileNotFoundError:
                logger.warning(
                    "Parameters file not found for node '%s': %s",
                    node_data.get("name"),
                    params_path,
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Invalid JSON in parameters file for node '%s': %s (%s)",
                    node_data.get("name"),
                    params_path,
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load parameters file for node '%s': %s (%s)",
                    node_data.get("name"),
                    params_path,
                    exc,
                )

        if param_defaults is not None:
            node_data["parameter_defaults"] = param_defaults

        try:
            templates.append(NodeTemplate(**node_data))
        except Exception as exc:
            raise RuntimeError(f"Invalid catalog format for node '{node_data.get('name')}': {exc}")

    return templates


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
