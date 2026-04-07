import os
import pandas as pd
import numpy as np
from openai import OpenAI
from data_cleaning_env.env import DataCleaningEnv, Action

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Prompt ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a data cleaning expert. You will be given a messy CSV dataset.
Your job is to clean it and return ONLY the cleaned CSV data, nothing else.
No explanations, no markdown, just the raw CSV text.

Common issues to fix:
- Missing values: fill numeric columns with column mean
- Duplicates: remove duplicate rows
- Wrong data types: convert to correct types
- Outliers: replace with column mean
- Inconsistent formats: standardize (dates, case, etc.)
- Invalid values: replace negatives with 0 or mean
"""

def get_llm_cleaning(dirty_csv: str, description: str) -> str:
    """Ask LLM to clean the dataset"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {description}\n\nDirty data:\n{dirty_csv}\n\nReturn only the cleaned CSV:"}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return ""

# ── Logging Helpers (MANDATORY FORMAT) ───────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    env = DataCleaningEnv()

    try:
        # Loop through each task and treat it as a brand new game
        for task in env.tasks:
            task_id = task["task_id"]
            
            # 1. Log START for this specific task
            log_start(task=task_id, env="data-cleaning-env", model=MODEL_NAME)

            # 2. Get the AI's cleaned data
            dirty_csv = task["dirty_df"].to_csv(index=False)
            cleaned_csv = get_llm_cleaning(dirty_csv, task["description"])

            action = Action(
                task_id=task_id,
                cleaned_data=cleaned_csv
            )

            # 3. Reset the environment and point it to the current task
            env.reset()
            env.current_task = task 
            
            # 4. Take the step
            obs, reward, done, info = env.step(action)

            # 5. Log the STEP
            action_name = f"clean_{task_id}_dataset"
            log_step(step=1, action=action_name, reward=reward.score, done=done, error=None)

            # 6. Log END for this specific task
            task_success = reward.score >= 0.45  # Passed threshold
            log_end(success=task_success, steps=1, score=reward.score, rewards=[reward.score])

    except Exception as e:
        print(f"[DEBUG] Error occurred: {e}")

if __name__ == "__main__":
    main()