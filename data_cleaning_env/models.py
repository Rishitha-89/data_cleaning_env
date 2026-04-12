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


class Reward(BaseModel):
    """Feedback after each action."""
    score: float                    # 0.0 to 1.0
    passed: bool                    # Did agent meet passing threshold?
    feedback: str                   # Human readable feedback
    improvement: float = 0.0       # Score improvement from last step