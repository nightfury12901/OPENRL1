"""
Pydantic Models for the Legal Case Assistant OpenEnv Environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LegalReward(BaseModel):
    total: float
    structural_score: float
    content_score: float
    length_penalty: float
    repetition_penalty: float
    step_bonus: float
    detail: str


class LegalAction(BaseModel):
    response: str = Field(..., description="The agent's complete response to the legal task.")


class LegalObservation(BaseModel):
    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    task_id: str = ""
    task_type: str = ""
    difficulty: str = ""
    prompt: str = ""
    input_text: str = ""
    expected_output_fields: List[str] = Field(default_factory=list)
    feedback: Optional[str] = None
    step_number: int = 0
    max_steps: int = 3
    reward_breakdown: Optional[Dict[str, Any]] = None


class LegalEnvironmentState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    current_task_index: int = 0
    total_steps: int = 0
    task_scores: Dict[str, float] = Field(default_factory=dict)
    task_feedbacks: Dict[str, str] = Field(default_factory=dict)
    done: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)
    previous_responses: List[str] = Field(default_factory=list)
