"""
Grading logic for the Data Cleaning Environment.

Each grader scores the agent's cleaned dataset against the expected output.
Scores are strictly between 0.0 and 1.0 (exclusive) with partial credit.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return max(0.01, min(round(score, 2), 0.99))


def score_easy_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Grade easy task: missing value imputation.
    
    Checks:
    - No missing values remain (0.30)
    - Age filled correctly with mean (0.25)
    - Salary filled correctly with mean (0.25)
    - Years exp filled (0.10)
    - Other columns unchanged (0.10)
    """
    score = 0.0

    try:
        # Check 1: No missing values remaining
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.30
        elif missing <= 2:
            score += 0.15  # Partial credit

        # Check 2: Age filled correctly (within 1.0 tolerance)
        try:
            expected_age = clean_df["age"].values.astype(float)
            agent_age = agent_df["age"].fillna(0).values.astype(float)
            matches = np.sum(np.abs(expected_age - agent_age) < 1.0)
            score += 0.25 * (matches / len(expected_age))
        except Exception:
            pass

        # Check 3: Salary filled correctly (within 500 tolerance)
        try:
            expected_sal = clean_df["salary"].values.astype(float)
            agent_sal = agent_df["salary"].fillna(0).values.astype(float)
            matches = np.sum(np.abs(expected_sal - agent_sal) < 500)
            score += 0.25 * (matches / len(expected_sal))
        except Exception:
            pass

        # Check 4: Years exp filled
        try:
            if agent_df["years_exp"].isnull().sum() == 0:
                score += 0.10
        except Exception:
            pass

        # Check 5: Name and department columns unchanged
        try:
            if (list(agent_df["name"]) == list(clean_df["name"]) and
                    list(agent_df["department"]) == list(clean_df["department"])):
                score += 0.10
        except Exception:
            pass

    except Exception:
        return 0.01

    return _clamp(score)


def score_medium_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Grade medium task: duplicates, type conversion, missing values.
    
    Checks:
    - Correct number of rows after dedup (0.25)
    - No missing values (0.20)
    - Age is numeric type (0.20)
    - Purchase amounts correct (0.20)
    - No invalid age values (0.15)
    """
    score = 0.0

    try:
        # Check 1: Duplicates removed — correct row count
        if len(agent_df) == len(clean_df):
            score += 0.25
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.10  # Partial credit

        # Check 2: No missing values
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.20
        elif missing <= 1:
            score += 0.10

        # Check 3: Age column is numeric
        try:
            if pd.api.types.is_numeric_dtype(agent_df["age"]):
                score += 0.20
            elif agent_df["age"].apply(lambda x: str(x).isdigit()).sum() > 3:
                score += 0.10  # Partial credit
        except Exception:
            pass

        # Check 4: Purchase amounts are correct
        try:
            expected = clean_df["purchase_amount"].values.astype(float)
            agent = agent_df["purchase_amount"].fillna(0).values.astype(float)
            if len(agent) == len(expected):
                matches = np.sum(np.abs(expected - agent) < 1.0)
                score += 0.20 * (matches / len(expected))
        except Exception:
            pass

        # Check 5: No invalid age values (all between 18-100)
        try:
            ages = pd.to_numeric(agent_df["age"], errors="coerce")
            valid = ages.between(18, 100).sum()
            score += 0.15 * (valid / len(ages))
        except Exception:
            pass

    except Exception:
        return 0.01

    return _clamp(score)


def score_hard_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    """
    Grade hard task: outliers, formats, case, invalid values.
    
    Checks:
    - Duplicates removed (0.15)
    - Product names in Title Case (0.15)
    - No negative stock (0.15)
    - Outlier prices replaced (0.15)
    - Category case consistent (0.15)
    - Dates in YYYY-MM-DD format (0.15)
    - Ratings within valid range (0.10)
    """
    score = 0.0

    try:
        # Check 1: Duplicates removed
        if len(agent_df) == len(clean_df):
            score += 0.15
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.07

        # Check 2: Product names in Title Case
        try:
            names = agent_df["product_name"].dropna().tolist()
            title_count = sum(1 for n in names if n == n.title())
            score += 0.15 * (title_count / len(names))
        except Exception:
            pass

        # Check 3: No negative stock values
        try:
            stock = agent_df["stock"].dropna()
            valid = (stock >= 0).sum()
            score += 0.15 * (valid / len(stock))
        except Exception:
            pass

        # Check 4: Outlier price (999.0) replaced
        try:
            prices = agent_df["price"].dropna()
            if prices.max() < 100:
                score += 0.15
            elif prices.max() < 500:
                score += 0.07
        except Exception:
            pass

        # Check 5: Category case consistent
        try:
            categories = agent_df["category"].dropna()
            unique_lower = set(c.lower() for c in categories)
            if len(unique_lower) == 1:
                score += 0.15
        except Exception:
            pass

        # Check 6: Dates in YYYY-MM-DD format
        try:
            dates = agent_df["date_added"].dropna()
            consistent = sum(
                1 for d in dates
                if str(d).count("-") == 2 and len(str(d)) == 10
            )
            score += 0.15 * (consistent / len(dates))
        except Exception:
            pass

        # Check 7: Ratings within valid range (1.0 - 5.0)
        try:
            ratings = agent_df["rating"].dropna()
            valid = ratings.between(1.0, 5.0).sum()
            score += 0.10 * (valid / len(ratings))
        except Exception:
            pass

    except Exception:
        return 0.01

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
        threshold = 0.70

    elif task_id == "medium":
        score = score_medium_task(agent_df, clean_df)
        threshold = 0.55

    elif task_id == "hard":
        score = score_hard_task(agent_df, clean_df)
        threshold = 0.45

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