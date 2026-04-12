"""
Pydantic models for the Data Cleaning Environment.
Defines the Observation, Action, and Reward types.
"""
from __future__ import annotations
from typing import List
from pydantic import BaseModel, field_validator


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    description: str
    difficulty: str
    issues: List[str]
    dirty_data: str
    step_count: int
    done: bool
    previous_score: float = 0.01  # Never 0.0!

    @field_validator("previous_score", mode="before")
    @classmethod
    def clamp_previous_score(cls, v):
        s = float(v)
        if s <= 0.0:
            return 0.01
        if s >= 1.0:
            return 0.99
        return round(s, 2)


class Action(BaseModel):
    """What the agent does — submit cleaned data."""
    task_id: str
    cleaned_data: str


class Reward(BaseModel):
    """Feedback after each action."""
    score: float
    passed: bool
    feedback: str
    improvement: float = 0.0

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v):
        """Strictly enforce score between 0 and 1 exclusive."""
        s = float(v)
        if s <= 0.0:
            return 0.01
        if s >= 1.0:
            return 0.99
        return round(s, 2)

    @field_validator("improvement", mode="before")
    @classmethod
    def clamp_improvement(cls, v):
        return round(float(v), 2)