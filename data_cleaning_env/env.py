import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple
from pydantic import BaseModel
from data_cleaning_env.tasks import get_all_tasks
from data_cleaning_env.graders import grade

class Observation(BaseModel):
    task_id: str
    description: str
    difficulty: str
    issues: list
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

class DataCleaningEnv:
    def __init__(self):
        self.tasks = get_all_tasks()
        self.current_task_idx = 0  # NEW: Tracks task rotation to show all 3 tasks to the bot
        self.current_task = None
        self.step_count = 0
        self.done = False
        self.max_steps = 10
        self.previous_score = 0.01
        self.best_score = 0.01

    def reset(self, task_id: str = None, **kwargs) -> Observation:
        # FIX: Allow the automated bot to discover all 3 tasks
        if task_id:
            self.current_task = next((t for t in self.tasks if t["task_id"] == task_id), self.tasks[0])
        else:
            self.current_task = self.tasks[self.current_task_idx]
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
            
        self.step_count = 0
        self.done = False
        self.previous_score = 0.01
        self.best_score = 0.01
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")
            
        self.step_count += 1
        task = next((t for t in self.tasks if t["task_id"] == action.task_id), None)
        
        if task is None:
            reward = Reward(score=0.01, passed=False, feedback="Invalid task_id", improvement=0.0)
            return self._make_observation(), reward, True, {}
            
        try:
            agent_df = pd.read_csv(pd.io.common.StringIO(action.cleaned_data))
        except Exception as e:
            reward = Reward(score=0.01, passed=False, feedback=f"Could not parse CSV: {e}", improvement=0.0)
            return self._make_observation(), reward, True, {}
            
        result = grade(action.task_id, agent_df, task["clean_df"])
        
        improvement = round(result["score"] - self.previous_score, 2)
        
        if self.step_count > 3 and improvement <= 0:
            result["score"] = max(0.01, result["score"] - 0.05)
            result["feedback"] += " ⚠️ Penalty: No improvement detected"
            
        self.previous_score = result["score"]
        self.best_score = max(self.best_score, result["score"])
        
        reward = Reward(
            score=result["score"],
            passed=result["passed"],
            feedback=result["feedback"],
            improvement=improvement
        )
        
        if self.step_count >= self.max_steps or reward.passed:
            self.done = True
            
        obs = self._make_observation()
        info = {
            "step": self.step_count,
            "task_id": action.task_id,
            "best_score": self.best_score
        }
        
        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.current_task["task_id"] if self.current_task else None,
            "step_count": self.step_count,
            "done": self.done,
            "total_tasks": len(self.tasks),
            "best_score": self.best_score,
            "previous_score": self.previous_score
        }

    def _make_observation(self) -> Observation:
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