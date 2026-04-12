from __future__ import annotations
import pandas as pd
from typing import Any, Dict, Tuple
from .models import Observation, Action, Reward
from .tasks import get_all_tasks
from .graders import grade


class DataCleaningEnv:
    def __init__(self):
        self.tasks = get_all_tasks()
        self.current_task = None
        self.current_task_idx = 0
        self.step_count = 0
        self.done = False
        self.max_steps = 10
        self.previous_score = 0.01
        self.best_score = 0.01

    def reset(self, task_id: str = None) -> Observation:
        if task_id:
            self.current_task = next(
                (t for t in self.tasks if t["task_id"] == task_id),
                self.tasks[0]
            )
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
            raise ValueError("Episode is done. Call reset() to start new episode.")

        self.step_count += 1

        task = next(
            (t for t in self.tasks if t["task_id"] == action.task_id),
            None
        )

        if task is None:
            reward = Reward(
                score=0.01,
                passed=False,
                feedback=f"Invalid task_id: {action.task_id}",
                improvement=0.0
            )
            self.done = True
            return self._make_observation(), reward, True, {}

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

        result = grade(action.task_id, agent_df, task["clean_df"])
        score = result["score"]

        improvement = round(score - self.previous_score, 2)

        if self.step_count > 3 and improvement <= 0:
            score = max(0.01, score - 0.05)
            result["feedback"] += " ⚠️ No improvement penalty"

        self.previous_score = score
        self.best_score = max(self.best_score, score)

        reward = Reward(
            score=score,
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
            "best_score": self.best_score,
            "improvement": improvement
        }

        return obs, reward, self.done, info

    def state(self) -> Dict[str, Any]:
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
        self.current_task = None
        self.done = True

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