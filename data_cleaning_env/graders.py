import pandas as pd
import numpy as np
from typing import Dict, Any

def score_easy_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        # Check 1: No missing values (0.30)
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.30
        elif missing <= 2:
            score += 0.15  # partial credit
            
        # Check 2: Age column filled correctly (0.25)
        try:
            expected_age = clean_df["age"].values
            agent_age = agent_df["age"].fillna(0).values
            matches = np.sum(np.abs(expected_age - agent_age) < 1.0)
            score += 0.25 * (matches / len(expected_age))
        except Exception:
            pass
            
        # Check 3: Salary column filled correctly (0.25)
        try:
            expected_sal = clean_df["salary"].values
            agent_sal = agent_df["salary"].fillna(0).values
            matches = np.sum(np.abs(expected_sal - agent_sal) < 500)
            score += 0.25 * (matches / len(expected_sal))
        except Exception:
            pass
            
        # Check 4: years_exp filled (0.10)
        try:
            if agent_df["years_exp"].isnull().sum() == 0:
                score += 0.10
        except Exception:
            pass
            
        # Check 5: Other columns unchanged (0.10)
        try:
            if list(agent_df["name"]) == list(clean_df["name"]) and \
               list(agent_df["department"]) == list(clean_df["department"]):
                score += 0.10
        except Exception:
            pass
    except Exception:
        return 0.01
        
    # STRICT CONSTRAINT FIX: Cap at 0.99, floor at 0.01
    return max(0.01, min(round(score, 2), 0.99))

def score_medium_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        # Check 1: Duplicates removed (0.25)
        if len(agent_df) == len(clean_df):
            score += 0.25
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.10  # partial credit
            
        # Check 2: No missing values (0.20)
        missing = agent_df.isnull().sum().sum()
        if missing == 0:
            score += 0.20
        elif missing <= 1:
            score += 0.10
            
        # Check 3: Age column is numeric (0.20)
        try:
            if pd.api.types.is_numeric_dtype(agent_df["age"]):
                score += 0.20
            elif agent_df["age"].apply(lambda x: str(x).isdigit()).sum() > 3:
                score += 0.10 # partial credit
        except Exception:
            pass
            
        # Check 4: Purchase amounts correct (0.20)
        try:
            expected = clean_df["purchase_amount"].values
            agent = agent_df["purchase_amount"].fillna(0).values
            if len(agent) == len(expected):
                matches = np.sum(np.abs(expected - agent) < 1.0)
                score += 0.20 * (matches / len(expected))
        except Exception:
            pass
            
        # Check 5: No invalid ages (0.15)
        try:
            ages = pd.to_numeric(agent_df["age"], errors="coerce")
            valid = ages.between(18, 100).sum()
            score += 0.15 * (valid / len(ages))
        except Exception:
            pass
    except Exception:
        return 0.01
        
    # STRICT CONSTRAINT FIX: Cap at 0.99, floor at 0.01
    return max(0.01, min(round(score, 2), 0.99))

def score_hard_task(agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> float:
    score = 0.0
    try:
        # Check 1: Duplicates removed (0.15)
        if len(agent_df) == len(clean_df):
            score += 0.15
        elif len(agent_df) <= len(clean_df) + 1:
            score += 0.07
            
        # Check 2: Product names in title case (0.15)
        try:
            names = agent_df["product_name"].dropna().tolist()
            title_count = sum(1 for n in names if n == n.title())
            score += 0.15 * (title_count / len(names))
        except Exception:
            pass
            
        # Check 3: No negative stock values (0.15)
        try:
            stock = agent_df["stock"].dropna()
            valid = (stock >= 0).sum()
            score += 0.15 * (valid / len(stock))
        except Exception:
            pass
            
        # Check 4: Outlier price removed/replaced (0.15)
        try:
            prices = agent_df["price"].dropna()
            if prices.max() < 100:
                score += 0.15
            elif prices.max() < 500:
                score += 0.07
        except Exception:
            pass
            
        # Check 5: Category column consistent (0.15)
        try:
            categories = agent_df["category"].dropna()
            unique_lower = set(c.lower() for c in categories)
            if len(unique_lower) == 1:
                score += 0.15
        except Exception:
            pass
            
        # Check 6: Dates in consistent format (0.15)
        try:
            dates = agent_df["date_added"].dropna()
            consistent = sum(1 for d in dates if str(d).count("-") == 2 and len(str(d)) == 10)
            score += 0.15 * (consistent / len(dates))
        except Exception:
            pass
            
        # Check 7: Rating outliers handled (0.10)
        try:
            ratings = agent_df["rating"].dropna()
            valid = ratings.between(1.0, 5.0).sum()
            score += 0.10 * (valid / len(ratings))
        except Exception:
            pass
    except Exception:
        return 0.01
        
    # STRICT CONSTRAINT FIX: Cap at 0.99, floor at 0.01
    return max(0.01, min(round(score, 2), 0.99))

def grade(task_id: str, agent_df: pd.DataFrame, clean_df: pd.DataFrame) -> Dict[str, Any]:
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
        return {"score": 0.01, "passed": False, "feedback": "Unknown task"}
        
    return {
        "score": score,
        "passed": score >= threshold,
        "feedback": f"Task '{task_id}' scored {score}/1.0 — {'Passed ✅' if score >= threshold else 'Failed ❌'}"
    }