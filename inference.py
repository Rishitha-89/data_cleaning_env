import os
from openai import OpenAI
from data_cleaning_env.env import DataCleaningEnv
from data_cleaning_env.models import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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


def clamp(score: float) -> float:
    """
    Strictly clamp score between 0 and 1 exclusive.
    Never returns exactly 0.0 or 1.0.
    """
    try:
        s = float(score)
        s = round(s, 2)
        if s <= 0.0:
            return 0.01
        if s >= 1.0:
            return 0.99
        return s
    except Exception:
        return 0.01


def get_llm_cleaning(dirty_csv: str, description: str) -> tuple:
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
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        return cleaned, None
    except Exception as e:
        return "", str(e)


def main():
    env = DataCleaningEnv()

    for task in env.tasks:
        task_id = task["task_id"]
        rewards = []
        steps = 0
        success = False
        last_error = None

        print(
            f"[START] task={task_id} env=data-cleaning-env model={MODEL_NAME}",
            flush=True
        )

        try:
            env.reset(task_id=task_id)

            dirty_csv = task["dirty_df"].to_csv(index=False)
            cleaned_csv, error = get_llm_cleaning(dirty_csv, task["description"])

            if error:
                last_error = error
                cleaned_csv = dirty_csv

            action = Action(task_id=task_id, cleaned_data=cleaned_csv)
            obs, reward, done, info = env.step(action)

            steps = 1

            # CLAMP HERE — before anything else
            clamped = clamp(reward.score)
            rewards.append(clamped)
            success = clamped >= 0.45

            print(
                f"[STEP] step={steps} "
                f"action=clean_{task_id}_dataset "
                f"reward={clamped:.2f} "
                f"done={str(done).lower()} "
                f"error={'null' if not last_error else last_error}",
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
            # CLAMP AGAIN — every single reward before printing
            safe_rewards = [clamp(r) for r in rewards]
            rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)

            # Double check — replace any 0.00 or 1.00 strings
            rewards_str = rewards_str.replace("0.00", "0.01").replace("1.00", "0.99")

            print(
                f"[END] success={str(success).lower()} "
                f"steps={steps} "
                f"rewards={rewards_str}",
                flush=True
            )

            env.close()
            env = DataCleaningEnv()


if __name__ == "__main__":
    main()