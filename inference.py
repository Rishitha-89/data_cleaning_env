"""
Baseline Inference Script for Data Cleaning Environment.

MANDATORY VARIABLES:
    API_BASE_URL  - API endpoint for the LLM (has default)
    MODEL_NAME    - Model identifier (has default)
    HF_TOKEN      - Hugging Face API token (required, no default)

OUTPUT FORMAT (strictly enforced):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> rewards=<r1,r2,...>
"""
import os
from openai import OpenAI
from data_cleaning_env.env import DataCleaningEnv
from data_cleaning_env.models import Action

# ── Environment Variables ─────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# HF_TOKEN is required — fail fast with clear message
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── OpenAI Client (mandatory per hackathon rules) ─────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ── System Prompt for Data Cleaning Agent ─────────────────────────────────────
SYSTEM_PROMPT = """You are an expert data cleaning agent.
You will receive a messy CSV dataset and must clean it.
Return ONLY the cleaned CSV data — no explanations, no markdown, no code blocks.
Just the raw CSV text starting with the header row.

Common issues to fix:
- Missing values: fill numeric columns with the column mean
- Duplicate rows: remove all duplicates, keep first occurrence
- Wrong data types: convert age/numeric columns to proper numbers
- Invalid values: replace "abc", "xyz" etc with column mean
- Outliers: replace extreme values (e.g. price=999) with column mean
- Negative values: replace negative stock/quantities with 0
- Inconsistent case: standardize text to Title Case
- Date formats: standardize all dates to YYYY-MM-DD format"""


def get_llm_cleaning(dirty_csv: str, description: str) -> tuple[str, str]:
    """
    Ask LLM to clean the dataset.
    
    Returns:
        Tuple of (cleaned_csv, error_message or None)
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task: {description}\n\n"
                        f"Dirty data:\n{dirty_csv}\n\n"
                        f"Return only the cleaned CSV:"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        cleaned = response.choices[0].message.content.strip()
        # Remove markdown code blocks if model added them
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        return cleaned, None
    except Exception as e:
        return "", str(e)


def main():
    """
    Run baseline agent against all 3 tasks.
    Outputs strictly formatted [START]/[STEP]/[END] lines.
    """
    env = DataCleaningEnv()
    all_rewards = []

    for task in env.tasks:
        task_id = task["task_id"]
        rewards = []
        steps = 0
        success = False
        last_error = None

        # ── [START] ───────────────────────────────────────────────────────────
        print(
            f"[START] task={task_id} env=data-cleaning-env model={MODEL_NAME}",
            flush=True
        )

        try:
            # Reset environment for this task
            env.reset(task_id=task_id)

            # Get dirty data and ask LLM to clean it
            dirty_csv = task["dirty_df"].to_csv(index=False)
            cleaned_csv, error = get_llm_cleaning(dirty_csv, task["description"])

            if error:
                last_error = error
                cleaned_csv = dirty_csv  # Fallback to dirty data

            # Submit cleaned data to environment
            action = Action(task_id=task_id, cleaned_data=cleaned_csv)
            obs, reward, done, info = env.step(action)

            steps = 1
            rewards.append(reward.score)
            success = reward.passed
            action_str = f"clean_{task_id}_dataset"

            # ── [STEP] ────────────────────────────────────────────────────────
            print(
                f"[STEP] step={steps} "
                f"action={action_str} "
                f"reward={reward.score:.2f} "
                f"done={str(done).lower()} "
                f"error={last_error if last_error else 'null'}",
                flush=True
            )

        except Exception as e:
            last_error = str(e)
            steps = max(steps, 1)
            rewards.append(0.01)
            print(
                f"[STEP] step={steps} "
                f"action=error "
                f"reward=0.01 "
                f"done=true "
                f"error={last_error}",
                flush=True
            )

        finally:
            # ── [END] ─────────────────────────────────────────────────────────
            env.close()
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={str(success).lower()} "
                f"steps={steps} "
                f"rewards={rewards_str}",
                flush=True
            )
            all_rewards.extend(rewards)

            # Reinitialize env for next task
            env = DataCleaningEnv()


if __name__ == "__main__":
    main()