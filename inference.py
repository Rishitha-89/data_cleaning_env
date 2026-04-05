import os
import pandas as pd
import numpy as np
from openai import OpenAI
from data_cleaning_env.env import DataCleaningEnv, Action

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    env = DataCleaningEnv()
    total_score = 0.0
    results = []

    print("🚀 Running baseline inference...\n")

    for task in env.tasks:
        print(f"📋 Task: {task['task_id']} ({task['difficulty']})")

        dirty_csv = task["dirty_df"].to_csv(index=False)
        cleaned_csv = get_llm_cleaning(dirty_csv, task["description"])

        action = Action(
            task_id=task["task_id"],
            cleaned_data=cleaned_csv
        )

        obs, reward, done, info = env.step(action) if env.current_task else (None, None, None, None)

        # Reset and step properly
        env.reset()
        obs, reward, done, info = env.step(action)

        print(f"   Score: {reward.score}")
        print(f"   Passed: {reward.passed}")
        print(f"   Feedback: {reward.feedback}\n")

        total_score += reward.score
        results.append({
            "task": task["task_id"],
            "score": reward.score,
            "passed": reward.passed
        })

    avg_score = total_score / len(env.tasks)
    print(f"✅ Average Score: {avg_score:.2f}")
    print(f"📊 Results: {results}")

if __name__ == "__main__":
    main()