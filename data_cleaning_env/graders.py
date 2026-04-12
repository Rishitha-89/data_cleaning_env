"""
Grading logic for the Data Cleaning Environment.

Each grader scores the agent's cleaned dataset against the expected output.
Scores are strictly between 0.0 and 1.0 (exclusive) with partial credit.
Maximum possible score is 0.97 — mathematically impossible to hit 1.0.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
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


def score_easy_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Grade easy task: missing value imputation.

    Weights are designed so max possible score = 0.97
    This makes it mathematically impossible to return exactly 1.0

    Checks:
    - No missing values remain (0.29)
    - Age filled correctly with mean (0.24)
    - Salary filled correctly with mean (0.24)
    - Years exp filled (0.10)
    - Other columns unchanged (0.10)
    Max possible = 0.97
    """
    score = 0.0

    try:
        # Check 1: No missing values remaining (0.29)
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.29
        elif missing <= 2:
            score += 0.14  # Partial credit

        # Check 2: Age filled correctly within 1.0 tolerance (0.24)
        try:
            expected_age = clean_df["age"].values.astype(float)
            agent_age = agent_df["age"].fillna(0).values.astype(float)
            matches = np.sum(np.abs(expected_age - agent_age) < 1.0)
            score += 0.24 * (matches / len(expected_age))
        except Exception:
            pass

        # Check 3: Salary filled correctly within 500 tolerance (0.24)
        try:
            expected_sal = clean_df["salary"].values.astype(float)
            agent_sal = agent_df["salary"].fillna(0).values.astype(float)
            matches = np.sum(np.abs(expected_sal - agent_sal) < 500)
            score += 0.24 * (matches / len(expected_sal))
        except Exception:
            pass

        # Check 4: Years exp filled (0.10)
        try:
            if agent_df["years_exp"].isnull().sum() == 0:
                score += 0.10
        except Exception:
            pass

        # Check 5: Name and department columns unchanged (0.10)
        try:
            if (list(agent_df["name"]) == list(clean_df["name"]) and
                    list(agent_df["department"]) == list(clean_df["department"])):
                score += 0.10
        except Exception:
            pass

    except Exception:
        return 0.01

    # Max possible = 0.29 + 0.24 + 0.24 + 0.10 + 0.10 = 0.97
    return _clamp(score)


def score_medium_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Grade medium task: duplicates, type conversion, missing values.

    Weights designed so max possible score = 0.97

    Checks:
    - Correct number of rows after dedup (0.24)
    - No missing values (0.19)
    - Age is numeric type (0.19)
    - Purchase amounts correct (0.19)
    - No invalid age values (0.16)
    Max possible = 0.97
    """
    score = 0.0

    try:
        # Check 1: Duplicates removed — correct row count (0.24)
        if len(agent_df) == len(clean_df):
            score += 0.24
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.10  # Partial credit

        # Check 2: No missing values (0.19)
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.19
        elif missing <= 1:
            score += 0.09

        # Check 3: Age column is numeric (0.19)
        try:
            if pd.api.types.is_numeric_dtype(agent_df["age"]):
                score += 0.19
            elif agent_df["age"].apply(lambda x: str(x).isdigit()).sum() > 3:
                score += 0.09  # Partial credit
        except Exception:
            pass

        # Check 4: Purchase amounts are correct (0.19)
        try:
            expected = clean_df["purchase_amount"].values.astype(float)
            agent = agent_df["purchase_amount"].fillna(0).values.astype(float)
            if len(agent) == len(expected):
                matches = np.sum(np.abs(expected - agent) < 1.0)
                score += 0.19 * (matches / len(expected))
        except Exception:
            pass

        # Check 5: No invalid age values between 18-100 (0.16)
        try:
            ages = pd.to_numeric(agent_df["age"], errors="coerce")
            valid = ages.between(18, 100).sum()
            score += 0.16 * (valid / len(ages))
        except Exception:
            pass

    except Exception:
        return 0.01

    # Max possible = 0.24 + 0.19 + 0.19 + 0.19 + 0.16 = 0.97
    return _clamp(score)


def score_hard_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Grade hard task: outliers, formats, case, invalid values.

    Weights designed so max possible score = 0.97

    Checks:
    - Duplicates removed (0.14)
    - Product names in Title Case (0.14)
    - No negative stock (0.14)
    - Outlier prices replaced (0.14)
    - Category case consistent (0.14)
    - Dates in YYYY-MM-DD format (0.14)
    - Ratings within valid range (0.13)
    Max possible = 0.97
    """
    score = 0.0

    try:
        # Check 1: Duplicates removed (0.14)
        if len(agent_df) == len(clean_df):
            score += 0.14
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.07

        # Check 2: Product names in Title Case (0.14)
        try:
            names = agent_df["product_name"].dropna().tolist()
            title_count = sum(1 for n in names if n == n.title())
            score += 0.14 * (title_count / len(names))
        except Exception:
            pass

        # Check 3: No negative stock values (0.14)
        try:
            stock = agent_df["stock"].dropna()
            valid = (stock >= 0).sum()
            score += 0.14 * (valid / len(stock))
        except Exception:
            pass

        # Check 4: Outlier price replaced (0.14)
        try:
            prices = agent_df["price"].dropna()
            if prices.max() < 100:
                score += 0.14
            elif prices.max() < 500:
                score += 0.07
        except Exception:
            pass

        # Check 5: Category case consistent (0.14)
        try:
            categories = agent_df["category"].dropna()
            unique_lower = set(c.lower() for c in categories)
            if len(unique_lower) == 1:
                score += 0.14
        except Exception:
            pass

        # Check 6: Dates in YYYY-MM-DD format (0.14)
        try:
            dates = agent_df["date_added"].dropna()
            consistent = sum(
                1 for d in dates
                if str(d).count("-") == 2 and len(str(d)) == 10
            )
            score += 0.14 * (consistent / len(dates))
        except Exception:
            pass

        # Check 7: Ratings within valid range 1.0-5.0 (0.13)
        try:
            ratings = agent_df["rating"].dropna()
            valid = ratings.between(1.0, 5.0).sum()
            score += 0.13 * (valid / len(ratings))
        except Exception:
            pass

    except Exception:
        return 0.01

    # Max possible = 0.14*6 + 0.13 = 0.97
    return _clamp(score)


def grade(task_id: str, agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main grader function — scores any task by ID.

    Args:
        task_id: "easy", "medium", or "hard"
        agent_df: The agent's cleaned DataFrame
        clean_df: The expected clean DataFrame

    Returns:
        Dict with score strictly between 0 and 1 (exclusive),
        passed (bool), and feedback (str)
    """
    if task_id == "easy":
        score = score_easy_task(agent_df, clean_df)
        threshold = 0.60

    elif task_id == "medium":
        score = score_medium_task(agent_df, clean_df)
        threshold = 0.50

    elif task_id == "hard":
        score = score_hard_task(agent_df, clean_df)
        threshold = 0.40

    else:
        return {
            "score": 0.01,
            "passed": False,
            "feedback": f"Unknown task_id: {task_id}"
        }

    # Final clamp — strictly between 0 and 1
    score = _clamp(score)
    passed = score >= threshold

    return {
        "score": score,
        "passed": passed,
        "feedback": (
            f"Task '{task_id}' scored {score:.2f}/1.0 — "
            f"{'Passed ✅' if passed else 'Failed ❌'} "
            f"(threshold: {threshold})"
        )
    }