from __future__ import annotations
from typing import List
from pydantic import BaseModel, field_validator


class Observation(BaseModel):
    task_id: str
    description: str
    difficulty: str
    issues: List[str]
    dirty_data: str
    step_count: int
    done: bool
    previous_score: float = 0.01


class Action(BaseModel):
    task_id: str
    cleaned_data: str


class Reward(BaseModel):
    score: float
    passed: bool
    feedback: str
    improvement: float = 0.0

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v):
        """Strictly enforce score between 0 and 1 exclusive."""
        return max(0.01, min(round(float(v), 2), 0.99))

    @field_validator("improvement", mode="before")
    @classmethod
    def clamp_improvement(cls, v):
        return round(float(v), 2)