"""
Pydantic models for the Data Cleaning Environment.
Defines the Observation, Action, and Reward types.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str                    # Which task: easy, medium, hard
    description: str                # Natural language task description
    difficulty: str                 # easy, medium, or hard
    issues: List[str]               # List of issues to fix
    dirty_data: str                 # CSV string of messy dataset
    step_count: int                 # How many steps taken so far
    done: bool                      # Is episode over?
    previous_score: float = 0.0    # Score from last step


class Action(BaseModel):
    """What the agent does — submit cleaned data."""
    task_id: str                    # Which task this action is for
    cleaned_data: str               # CSV string of cleaned dataset


from pydantic import BaseModel, field_validator

class Reward(BaseModel):
    """Feedback after each action."""
    score: float
    passed: bool
    feedback: str
    improvement: float = 0.0

    @field_validator("score")
    @classmethod
    def score_must_be_strictly_between_0_and_1(cls, v):
        """Strictly enforce score is between 0 and 1 exclusive."""
        return max(0.01, min(round(v, 2), 0.99))