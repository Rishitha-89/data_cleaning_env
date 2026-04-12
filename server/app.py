"""
FastAPI server for the Data Cleaning Environment.

Exposes OpenEnv standard endpoints:
- GET  /       → health check
- POST /reset  → start new episode
- POST /step   → submit cleaned data
- GET  /state  → current environment state
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure parent directory is in path for imports
SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from data_cleaning_env.env import DataCleaningEnv
from data_cleaning_env.models import Action

# Initialize FastAPI app
app = FastAPI(
    title="Data Cleaning Environment",
    description="OpenEnv environment for training AI agents to clean messy datasets",
    version="1.0.0"
)

# Single shared environment instance
env = DataCleaningEnv()


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Data Cleaning Environment is running!",
        "tasks": ["easy", "medium", "hard"]
    }


@app.post("/reset")
def reset(task_id: str = None):
    """
    Reset the environment and get initial observation.
    
    Args:
        task_id: Optional specific task (easy/medium/hard).
                 Rotates automatically if not specified.
    """
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
            "reward": reward.score,  # ← Return float directly!
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
    """Get current environment state."""
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main(host: str = "0.0.0.0", port: int = None):
    """Run the Data Cleaning Environment server."""
    import uvicorn
    if port is None:
        port = int(os.getenv("API_PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()