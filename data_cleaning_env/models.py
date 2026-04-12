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
    previous_score: float = 0.001

    @field_validator("previous_score", mode="before")
    @classmethod
    def clamp_previous_score(cls, v):
        s = float(v)
        if s <= 0.0:
            return float(0.001)
        if s >= 1.0:
            return float(0.99)
        return float(round(s, 3))


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
        s = float(v)
        if s <= 0.0:
            return float(0.001)
        if s >= 1.0:
            return float(0.99)
        return float(round(s, 3))

    @field_validator("improvement", mode="before")
    @classmethod
    def clamp_improvement(cls, v):
        return float(round(float(v), 3))