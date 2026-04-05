import pandas as pd
import numpy as np
from typing import Dict, Any


def score_easy_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Scores the agent's cleaning of the easy task.
    Checks if missing values are filled correctly.
    Returns score between 0.0 and 1.0
    """
    score = 0.0
    total_checks = 4

    try:
        # Check 1: No missing values remaining (0.25)
        if agent_df.isnull().sum().sum() == 0:
            score += 0.25

        # Check 2: Age column filled correctly (0.25)
        expected_age = clean_df["age"].values
        agent_age = agent_df["age"].values
        if np.allclose(expected_age, agent_age, atol=0.5):
            score += 0.25

        # Check 3: Salary column filled correctly (0.25)
        expected_salary = clean_df["salary"].values
        agent_salary = agent_df["salary"].values
        if np.allclose(expected_salary, agent_salary, atol=100):
            score += 0.25

        # Check 4: Other columns unchanged (0.25)
        if list(agent_df["name"]) == list(clean_df["name"]) and \
           list(agent_df["department"]) == list(clean_df["department"]):
            score += 0.25

    except Exception:
        return 0.0

    return round(score, 2)


def score_medium_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Scores the agent's cleaning of the medium task.
    Checks duplicates removed, types fixed, missing values filled.
    Returns score between 0.0 and 1.0
    """
    score = 0.0

    try:
        # Check 1: Duplicates removed (0.30)
        if len(agent_df) == len(clean_df):
            score += 0.30

        # Check 2: No missing values (0.25)
        if agent_df.isnull().sum().sum() == 0:
            score += 0.25

        # Check 3: Age column is numeric (0.20)
        if pd.api.types.is_numeric_dtype(agent_df["age"]):
            score += 0.20

        # Check 4: Purchase amounts correct (0.25)
        try:
            expected = clean_df["purchase_amount"].values
            agent = agent_df["purchase_amount"].values
            if np.allclose(expected, agent, atol=1.0):
                score += 0.25
        except Exception:
            pass

    except Exception:
        return 0.0

    return round(score, 2)


def score_hard_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Scores the agent's cleaning of the hard task.
    Checks duplicates, outliers, formats, invalid values, case consistency.
    Returns score between 0.0 and 1.0
    """
    score = 0.0

    try:
        # Check 1: Duplicates removed (0.20)
        if len(agent_df) == len(clean_df):
            score += 0.20

        # Check 2: Product names in title case (0.20)
        try:
            names = agent_df["product_name"].dropna().tolist()
            if all(n == n.title() for n in names):
                score += 0.20
        except Exception:
            pass

        # Check 3: No negative stock values (0.20)
        try:
            stock = agent_df["stock"].dropna()
            if (stock >= 0).all():
                score += 0.20
        except Exception:
            pass

        # Check 4: Outlier price removed/replaced (0.20)
        try:
            prices = agent_df["price"].dropna()
            if prices.max() < 100:  # 999 was the outlier
                score += 0.20
        except Exception:
            pass

        # Check 5: Category column consistent case (0.20)
        try:
            categories = agent_df["category"].dropna().unique()
            if len(set(c.lower() for c in categories)) == 1:
                score += 0.20
        except Exception:
            pass

    except Exception:
        return 0.0

    return round(score, 2)


def grade(task_id: str, agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main grader function — call this to score any task.
    Returns a dict with score and feedback.
    """
    if task_id == "easy":
        score = score_easy_task(agent_df, clean_df)
        threshold = 0.75

    elif task_id == "medium":
        score = score_medium_task(agent_df, clean_df)
        threshold = 0.60

    elif task_id == "hard":
        score = score_hard_task(agent_df, clean_df)
        threshold = 0.50

    else:
        return {"score": 0.0, "passed": False, "feedback": "Unknown task"}

    return {
        "score": score,
        "passed": score >= threshold,
        "feedback": f"Task '{task_id}' scored {score}/1.0 — {'Passed ✅' if score >= threshold else 'Failed ❌'}"
    }