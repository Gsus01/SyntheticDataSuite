from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ComponentType = Literal[
    "input",
    "preprocessing",
    "training",
    "generation",
    "output",
    "other",
]


class Port(BaseModel):
    name: str = Field(..., description="Short port name (kebab-case recommended).")
    path: str = Field(
        ..., description="Absolute path inside container (e.g., /data/inputs/...)"
    )
    role: Optional[str] = Field(
        None, description="Optional role: data|config|model|metrics|other"
    )


class ComponentPlan(BaseModel):
    name: str = Field(..., description="Component identifier (kebab-case).")
    title: str = Field(..., description="Human-readable title.")
    type: ComponentType = Field(..., description="Component category.")
    description: str = Field(..., description="What this component does.")
    inputs: List[Port] = Field(default_factory=list)
    outputs: List[Port] = Field(default_factory=list)
    parameters_defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameters shown in the inspector (defaults only).",
    )
    notes: List[str] = Field(default_factory=list)


class ExtractionPlan(BaseModel):
    components: List[ComponentPlan]
    rationale: str = Field(..., description="Why these components were chosen.")
    assumptions: List[str] = Field(default_factory=list)


class GeneratedComponentFiles(BaseModel):
    main_py: str
    dockerfile: str
    componentspec: Dict[str, Any]
    requirements_txt: str = ""
    readme_md: str = ""


class ReviewIssue(BaseModel):
    component: str
    message: str


class ReviewReport(BaseModel):
    status: Literal["OK", "NEEDS_FIX"]
    issues: List[ReviewIssue] = Field(default_factory=list)
