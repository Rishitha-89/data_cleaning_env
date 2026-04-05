import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
from pydantic import BaseModel
from data_cleaning_env.tasks import get_all_tasks
from data_cleaning_env.graders import grade


# ── Typed Models (OpenEnv spec) ──────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    description: str
    difficulty: str
    issues: list
    dirty_data: str        # CSV string of messy data
    step_count: int
    done: bool

class Action(BaseModel):
    task_id: str
    cleaned_data: str      # CSV string of cleaned data

class Reward(BaseModel):
    score: float
    passed: bool
    feedback: str


# ── Main Environment ─────────────────────────────────────────────────────────

class DataCleaningEnv:
    def __init__(self):
        self.tasks = get_all_tasks()
        self.current_task = None
        self.step_count = 0
        self.done = False
        self.max_steps = 10

    def reset(self) -> Observation:
        """Reset environment to first task"""
        self.current_task = self.tasks[0]
        self.step_count = 0
        self.done = False
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """Take a cleaning action and return results"""
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        self.step_count += 1

        # Find the matching task
        task = next((t for t in self.tasks if t["task_id"] == action.task_id), None)
        if task is None:
            reward = Reward(score=0.0, passed=False, feedback="Invalid task_id")
            return self._make_observation(), reward, True, {}

        # Parse agent's cleaned data
        try:
            agent_df = pd.read_csv(pd.io.common.StringIO(action.cleaned_data))
        except Exception as e:
            reward = Reward(score=0.0, passed=False, feedback=f"Could not parse CSV: {e}")
            return self._make_observation(), reward, True, {}

        # Grade the submission
        result = grade(action.task_id, agent_df, task["clean_df"])
        reward = Reward(**result)

        # Check if done
        if self.step_count >= self.max_steps or reward.passed:
            self.done = True

        obs = self._make_observation()
        info = {"step": self.step_count, "task_id": action.task_id}

        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return current state"""
        return {
            "task_id": self.current_task["task_id"] if self.current_task else None,
            "step_count": self.step_count,
            "done": self.done,
            "total_tasks": len(self.tasks)
        }

    def _make_observation(self) -> Observation:
        """Helper to build observation from current task"""
        task = self.current_task
        return Observation(
            task_id=task["task_id"],
            description=task["description"],
            difficulty=task["difficulty"],
            issues=task["issues"],
            dirty_data=task["dirty_df"].to_csv(index=False),
            step_count=self.step_count,
            done=self.done
        )