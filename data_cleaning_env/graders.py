from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any


def _clamp(score: float) -> float:
    """Score must be strictly between 0 and 1 exclusive. Always returns float."""
    try:
        s = float(score)
        s = round(s, 2)
        if s <= 0.0:
            return float(0.01)
        if s >= 1.0:
            return float(0.99)
        return float(s)
    except Exception:
        return float(0.01)


def score_easy_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        # Check 1: No missing values (0.29)
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.29
        elif missing <= 2:
            score += 0.14

        # Check 2: Age filled correctly (0.24)
        try:
            expected_age = clean_df["age"].values.astype(float)
            agent_age = agent_df["age"].fillna(0).values.astype(float)
            matches = np.sum(np.abs(expected_age - agent_age) < 1.0)
            score += 0.24 * (matches / len(expected_age))
        except Exception:
            pass

        # Check 3: Salary filled correctly (0.24)
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

        # Check 5: Other columns unchanged (0.10)
        try:
            if (list(agent_df["name"]) == list(clean_df["name"]) and
                    list(agent_df["department"]) == list(clean_df["department"])):
                score += 0.10
        except Exception:
            pass

    except Exception:
        return float(0.01)

    return _clamp(score)


def score_medium_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        # Check 1: Duplicates removed (0.24)
        if len(agent_df) == len(clean_df):
            score += 0.24
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.10

        # Check 2: No missing values (0.19)
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.19
        elif missing <= 1:
            score += 0.09

        # Check 3: Age is numeric (0.19)
        try:
            if pd.api.types.is_numeric_dtype(agent_df["age"]):
                score += 0.19
            elif agent_df["age"].apply(lambda x: str(x).isdigit()).sum() > 3:
                score += 0.09
        except Exception:
            pass

        # Check 4: Purchase amounts correct (0.19)
        try:
            expected = clean_df["purchase_amount"].values.astype(float)
            agent = agent_df["purchase_amount"].fillna(0).values.astype(float)
            if len(agent) == len(expected):
                matches = np.sum(np.abs(expected - agent) < 1.0)
                score += 0.19 * (matches / len(expected))
        except Exception:
            pass

        # Check 5: No invalid ages (0.16)
        try:
            ages = pd.to_numeric(agent_df["age"], errors="coerce")
            valid = ages.between(18, 100).sum()
            score += 0.16 * (float(valid) / float(len(ages)))
        except Exception:
            pass

    except Exception:
        return float(0.01)

    return _clamp(score)


def score_hard_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        # Check 1: Duplicates removed (0.14)
        if len(agent_df) == len(clean_df):
            score += 0.14
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.07

        # Check 2: Product names Title Case (0.14)
        try:
            names = agent_df["product_name"].dropna().tolist()
            title_count = sum(1 for n in names if n == n.title())
            score += 0.14 * (float(title_count) / float(len(names)))
        except Exception:
            pass

        # Check 3: No negative stock (0.14)
        try:
            stock = agent_df["stock"].dropna()
            valid = (stock >= 0).sum()
            score += 0.14 * (float(valid) / float(len(stock)))
        except Exception:
            pass

        # Check 4: Outlier price replaced (0.14)
        try:
            prices = agent_df["price"].dropna()
            if float(prices.max()) < 100:
                score += 0.14
            elif float(prices.max()) < 500:
                score += 0.07
        except Exception:
            pass

        # Check 5: Category consistent (0.14)
        try:
            categories = agent_df["category"].dropna()
            unique_lower = set(c.lower() for c in categories)
            if len(unique_lower) == 1:
                score += 0.14
        except Exception:
            pass

        # Check 6: Dates in YYYY-MM-DD (0.14)
        try:
            dates = agent_df["date_added"].dropna()
            consistent = sum(
                1 for d in dates
                if str(d).count("-") == 2 and len(str(d)) == 10
            )
            score += 0.14 * (float(consistent) / float(len(dates)))
        except Exception:
            pass

        # Check 7: Ratings 1.0-5.0 (0.13)
        try:
            ratings = agent_df["rating"].dropna()
            valid = ratings.between(1.0, 5.0).sum()
            score += 0.13 * (float(valid) / float(len(ratings)))
        except Exception:
            pass

    except Exception:
        return float(0.01)

    return _clamp(score)


def grade(task_id: str, agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
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
            "score": float(0.01),
            "passed": False,
            "feedback": f"Unknown task_id: {task_id}"
        }

    score = _clamp(score)
    passed = bool(score >= threshold)

    return {
        "score": float(score),
        "passed": passed,
        "feedback": (
            f"Task '{task_id}' scored {score:.2f}/1.0 — "
            f"{'Passed ✅' if passed else 'Failed ❌'} "
            f"(threshold: {threshold})"
        )
    }