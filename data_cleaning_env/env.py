"""
Core Data Cleaning Environment.

Implements the OpenEnv interface:
- reset() → initial observation
- step(action) → observation, reward, done, info  
- state() → current environment state
- close() → cleanup
"""
from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Tuple

from .models import Observation, Action, Reward
from .tasks import get_all_tasks
from .graders import grade


class DataCleaningEnv:
    """
    A real-world RL environment for training AI agents to clean messy datasets.
    
    The agent receives dirty datasets and must return cleaned versions.
    Rewards are based on how well the agent cleans the data.
    """

    def __init__(self):
        # Load all 3 tasks on initialization
        self.tasks = get_all_tasks()
        self.current_task = None
        self.current_task_idx = 0      # Rotates through tasks on each reset
        self.step_count = 0
        self.done = False
        self.max_steps = 10            # Max steps before episode ends
        self.previous_score = 0.0
        self.best_score = 0.0

    def reset(self, task_id: str = None) -> Observation:
        """
        Reset the environment to start a new episode.
        
        Args:
            task_id: Optional specific task to start with.
                     If None, rotates through tasks in order.
        
        Returns:
            Initial observation with dirty dataset
        """
        # Allow specific task selection or rotate through tasks
        if task_id:
            self.current_task = next(
                (t for t in self.tasks if t["task_id"] == task_id),
                self.tasks[0]
            )
        else:
            self.current_task = self.tasks[self.current_task_idx]
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)

        # Reset episode state
        self.step_count = 0
        self.done = False
        self.previous_score = 0.0
        self.best_score = 0.0

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Take a cleaning action and receive feedback.
        
        Args:
            action: Contains task_id and cleaned_data (CSV string)
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() to start new episode.")

        self.step_count += 1

        # Find the task this action is for
        task = next(
            (t for t in self.tasks if t["task_id"] == action.task_id),
            None
        )

        # Invalid task ID
        if task is None:
            reward = Reward(
                score=0.01,
                passed=False,
                feedback=f"Invalid task_id: {action.task_id}",
                improvement=0.0
            )
            self.done = True
            return self._make_observation(), reward, True, {}

        # Parse agent's cleaned CSV
        try:
            agent_df = pd.read_csv(pd.io.common.StringIO(action.cleaned_data))
        except Exception as e:
            reward = Reward(
                score=0.01,
                passed=False,
                feedback=f"Could not parse cleaned CSV: {e}",
                improvement=0.0
            )
            self.done = True
            return self._make_observation(), reward, True, {}

        # Grade the submission
        result = grade(action.task_id, agent_df, task["clean_df"])

        # Calculate improvement over previous attempt
        improvement = round(result["score"] - self.previous_score, 2)

        # Penalize if agent makes no progress after 3 steps (anti-gaming)
        if self.step_count > 3 and improvement <= 0:
            result["score"] = max(0.01, result["score"] - 0.05)
            result["feedback"] += " ⚠️ No improvement penalty applied"

        # Update state
        self.previous_score = result["score"]
        self.best_score = max(self.best_score, result["score"])

        reward = Reward(
            score=result["score"],
            passed=result["passed"],
            feedback=result["feedback"],
            improvement=improvement
        )

        # End episode if max steps reached or task passed
        if self.step_count >= self.max_steps or reward.passed:
            self.done = True

        obs = self._make_observation()
        info = {
            "step": self.step_count,
            "task_id": action.task_id,
            "best_score": self.best_score,
            "improvement": improvement
        }

        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "task_id": self.current_task["task_id"] if self.current_task else None,
            "step_count": self.step_count,
            "done": self.done,
            "total_tasks": len(self.tasks),
            "best_score": self.best_score,
            "previous_score": self.previous_score,
            "available_tasks": [t["task_id"] for t in self.tasks]
        }

    def close(self) -> None:
        """Cleanup environment resources."""
        self.current_task = None
        self.done = True

    def _make_observation(self) -> Observation:
        """Build observation from current task state."""
        task = self.current_task
        return Observation(
            task_id=task["task_id"],
            description=task["description"],
            difficulty=task["difficulty"],
            issues=task["issues"],
            dirty_data=task["dirty_df"].to_csv(index=False),
            step_count=self.step_count,
            done=self.done,
            previous_score=self.previous_score
        )