from __future__ import annotations
import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from data_cleaning_env.env import DataCleaningEnv
from data_cleaning_env.models import Action

app = FastAPI(
    title="Data Cleaning Environment",
    description="OpenEnv environment for training AI agents to clean messy datasets",
    version="1.0.0"
)

env = DataCleaningEnv()


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Data Cleaning Environment is running!",
        "tasks": ["easy", "medium", "hard"]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: str = None):
    try:
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.score,
            "done": done,
            "info": info,
            "feedback": reward.feedback,
            "passed": reward.passed
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "issues": t["issues"]
            }
            for t in env.tasks
        ]
    }


def main(host: str = "0.0.0.0", port: int = None):
    import uvicorn
    if port is None:
        port = int(os.getenv("API_PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()