from __future__ import annotations
import pandas as pd


def get_easy_task() -> dict:
    dirty_data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve",
                 "Frank", "Grace", "Henry", "Iris", "Jack"],
        "age": [25, None, 30, None, 28, 35, None, 42, 29, None],
        "salary": [50000, 60000, None, 55000, None, 70000, 65000, None, 52000, 58000],
        "department": ["HR", "IT", "IT", "HR", "Finance",
                       "IT", "HR", "Finance", "IT", "HR"],
        "years_exp": [2, 5, None, 3, None, 8, 6, None, 4, 3]
    }
    clean_data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve",
                 "Frank", "Grace", "Henry", "Iris", "Jack"],
        "age": [25, 31.5, 30, 31.5, 28, 35, 31.5, 42, 29, 31.5],
        "salary": [50000, 60000, 58333, 55000, 58333, 70000, 65000, 58333, 52000, 58000],
        "department": ["HR", "IT", "IT", "HR", "Finance",
                       "IT", "HR", "Finance", "IT", "HR"],
        "years_exp": [2, 5, 4.8, 3, 4.8, 8, 6, 4.8, 4, 3]
    }
    return {
        "task_id": "easy",
        "description": (
            "Fix missing values in this employee HR dataset. "
            "Fill missing numeric values (age, salary, years_exp) with their column means. "
            "Do not modify name or department columns."
        ),
        "dirty_df": pd.DataFrame(dirty_data),
        "clean_df": pd.DataFrame(clean_data),
        "difficulty": "easy",
        "issues": ["missing_values"]
    }


def get_medium_task() -> dict:
    dirty_data = {
        "customer_id": [101, 102, 102, 103, 104, 104, 105, 106, 107, 107],
        "name": ["John", "Jane", "Jane", "Bob", "Alice", "Alice",
                 "Charlie", "Diana", "Eve", "Eve"],
        "age": ["25", "30", "30", "abc", "28", "28", "35", "xyz", "29", "29"],
        "purchase_amount": [200.5, None, 150.0, 300.0, None, 250.0,
                            175.0, 400.0, None, 320.0],
        "email": ["john@mail.com", "jane@mail.com", "jane@mail.com",
                  "bob@mail.com", "alice@mail.com", "alice@mail.com",
                  "charlie@mail.com", "diana@mail.com", "eve@mail.com", "eve@mail.com"],
        "city": ["NYC", "LA", "LA", "NYC", "Chicago", "Chicago",
                 "LA", "NYC", "Chicago", "Chicago"]
    }
    clean_data = {
        "customer_id": [101, 102, 103, 104, 105, 106, 107],
        "name": ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 29, 28, 35, 29, 29],
        "purchase_amount": [200.5, 150.0, 300.0, 250.0, 175.0, 400.0, 320.0],
        "email": ["john@mail.com", "jane@mail.com", "bob@mail.com",
                  "alice@mail.com", "charlie@mail.com", "diana@mail.com", "eve@mail.com"],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago"]
    }
    return {
        "task_id": "medium",
        "description": (
            "Fix this customer dataset. "
            "Remove duplicate rows, convert age column to numeric "
            "(replace invalid values with mean), "
            "and fill missing purchase_amount values with the column mean."
        ),
        "dirty_df": pd.DataFrame(dirty_data),
        "clean_df": pd.DataFrame(clean_data),
        "difficulty": "medium",
        "issues": ["duplicates", "wrong_types", "missing_values"]
    }


def get_hard_task() -> dict:
    dirty_data = {
        "product_id": [1, 2, 2, 3, 4, 5, 6, 7, 8, 8],
        "product_name": ["Apple", "banana", "banana", "CHERRY",
                         "Date", "elderberry", "Fig", "GRAPE", "honeydew", "honeydew"],
        "price": [1.5, 0.5, 0.5, 2.0, 999.0, 3.0, None, 4.0, 2.5, 2.5],
        "date_added": ["2024-01-15", "15/02/2024", "15/02/2024",
                       "2024-03-20", "2024-04-01", "05-05-2024",
                       None, "2024-07-10", "10/08/2024", "10/08/2024"],
        "stock": [100, 200, 200, None, 50, -10, 80, -5, 150, 150],
        "category": ["Fruit", "fruit", "fruit", "Fruit",
                     "fruit", "Fruit", "fruit", "FRUIT", "Fruit", "Fruit"],
        "rating": [4.5, 3.8, 3.8, 4.2, 1.0, 4.7, None, 4.1, 3.9, 3.9]
    }
    clean_data = {
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "product_name": ["Apple", "Banana", "Cherry", "Date",
                         "Elderberry", "Fig", "Grape", "Honeydew"],
        "price": [1.5, 0.5, 2.0, 2.5, 3.0, 2.5, 4.0, 2.5],
        "date_added": ["2024-01-15", "2024-02-15", "2024-03-20",
                       "2024-04-01", "2024-05-05", None, "2024-07-10", "2024-08-10"],
        "stock": [100, 200, None, 50, 0, 80, 0, 150],
        "category": ["Fruit", "Fruit", "Fruit", "Fruit",
                     "Fruit", "Fruit", "Fruit", "Fruit"],
        "rating": [4.5, 3.8, 4.2, 4.0, 4.7, 4.0, 4.1, 3.9]
    }
    return {
        "task_id": "hard",
        "description": (
            "Fix this product catalog dataset. "
            "Remove duplicates, standardize product names to Title Case, "
            "replace outlier prices (>100) with column mean, "
            "standardize all dates to YYYY-MM-DD format, "
            "replace negative stock values with 0, "
            "standardize category to Title Case, "
            "and ensure ratings are between 1.0 and 5.0."
        ),
        "dirty_df": pd.DataFrame(dirty_data),
        "clean_df": pd.DataFrame(clean_data),
        "difficulty": "hard",
        "issues": ["duplicates", "outliers", "inconsistent_formats",
                   "invalid_values", "missing_values", "inconsistent_case"]
    }


def get_all_tasks() -> list:
    return [get_easy_task(), get_medium_task(), get_hard_task()]