"""
Task definitions for the Data Cleaning Environment.
Three real-world inspired datasets with increasing difficulty.
Upgraded to include edge-case injection, corrupted types, and extreme outliers
to challenge state-of-the-art frontier models.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def get_easy_task() -> dict:
    # 20 Rows: Concentrated Missing Value Chaos
    dirty_data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
                 "Karen", "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ryan", "Sophia", "Tom"],
        "age": [25, None, 30, None, 28, 35, None, 42, 29, None, 
                22, 45, None, 31, 38, None, 26, 33, None, 27],
        "salary": [50000, 60000, None, 55000, None, 70000, 65000, None, 52000, 58000,
                   48000, 80000, None, 62000, 58000, 54000, 66000, None, 71000, 49000],
        "department": ["HR", "IT", "IT", "HR", "Finance", "IT", "HR", "Finance", "IT", "HR",
                       "Sales", "IT", "Finance", "HR", "Sales", "IT", "Finance", "HR", "IT", "Sales"],
        "years_exp": [2, 5, None, 3, None, 8, 6, None, 4, 3,
                      1, 10, 5, None, 7, 3, 6, None, 8, 2]
    }
    
    # Mathematical Means: Age (30.8), Salary (59866.66), Years_Exp (4.86)
    clean_data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
                 "Karen", "Liam", "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ryan", "Sophia", "Tom"],
        "age": [25, 30.8, 30, 30.8, 28, 35, 30.8, 42, 29, 30.8, 
                22, 45, 30.8, 31, 38, 30.8, 26, 33, 30.8, 27],
        "salary": [50000, 60000, 59866.66, 55000, 59866.66, 70000, 65000, 59866.66, 52000, 58000,
                   48000, 80000, 59866.66, 62000, 58000, 54000, 66000, 59866.66, 71000, 49000],
        "department": ["HR", "IT", "IT", "HR", "Finance", "IT", "HR", "Finance", "IT", "HR",
                       "Sales", "IT", "Finance", "HR", "Sales", "IT", "Finance", "HR", "IT", "Sales"],
        "years_exp": [2, 5, 4.86, 3, 4.86, 8, 6, 4.86, 4, 3,
                      1, 10, 5, 4.86, 7, 3, 6, 4.86, 8, 2]
    }
    
    return {
        "task_id": "easy",
        "description": (
            "Fix missing values in this employee HR dataset. "
            "Fill missing numeric values (age, salary, years_exp) with their exact column means. "
            "Do not modify name or department columns."
        ),
        "dirty_df": pd.DataFrame(dirty_data),
        "clean_df": pd.DataFrame(clean_data),
        "difficulty": "easy",
        "issues": ["missing_values"]
    }

def get_medium_task() -> dict:
    # 20 Rows: Duplicates and severe Type Mismatches (Text in Number columns)
    dirty_data = {
        "customer_id": [101, 102, 102, 103, 104, 104, 105, 106, 107, 107,
                        108, 109, 110, 110, 111, 112, 113, 114, 115, 115],
        "name": ["John", "Jane", "Jane", "Bob", "Alice", "Alice", "Charlie", "Diana", "Eve", "Eve",
                 "Frank", "Grace", "Hank", "Hank", "Ivy", "Jack", "Kim", "Leo", "Mia", "Mia"],
        "age": ["25", "30", "30", "abc", "28", "28", "35", "xyz", "29", "29",
                "Twenty", "32", "None", "None", "27", "40 years", "31", "NaN", "26", "26"],
        "purchase_amount": [200.5, None, 150.0, 300.0, None, 250.0, 175.0, 400.0, None, 320.0,
                            100.0, 500.0, 225.0, 225.0, None, 350.0, 275.0, 450.0, None, 125.0],
        "email": ["john@mail.com", "jane@mail.com", "jane@mail.com", "bob@mail.com", "alice@mail.com", "alice@mail.com",
                  "charlie@mail.com", "diana@mail.com", "eve@mail.com", "eve@mail.com", "frank@mail.com", "grace@mail.com",
                  "hank@mail.com", "hank@mail.com", "ivy@mail.com", "jack@mail.com", "kim@mail.com", "leo@mail.com", "mia@mail.com", "mia@mail.com"],
        "city": ["NYC", "LA", "LA", "NYC", "Chicago", "Chicago", "LA", "NYC", "Chicago", "Chicago",
                 "Miami", "NYC", "LA", "LA", "Chicago", "Miami", "NYC", "LA", "Chicago", "Chicago"]
    }
    
    clean_data = {
        "customer_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
        "name": ["John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Kim", "Leo", "Mia"],
        "age": [25, 30, 30.6, 28, 35, 30.6, 29, 30.6, 32, 30.6, 27, 40, 31, 30.6, 26],
        "purchase_amount": [200.5, 150.0, 300.0, 250.0, 175.0, 400.0, 320.0, 100.0, 500.0, 225.0, 282.9, 350.0, 275.0, 450.0, 125.0],
        "email": ["john@mail.com", "jane@mail.com", "bob@mail.com", "alice@mail.com", "charlie@mail.com", "diana@mail.com", "eve@mail.com", 
                  "frank@mail.com", "grace@mail.com", "hank@mail.com", "ivy@mail.com", "jack@mail.com", "kim@mail.com", "leo@mail.com", "mia@mail.com"],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "Miami", "NYC", "LA", "Chicago", "Miami", "NYC", "LA", "Chicago"]
    }
    
    return {
        "task_id": "medium",
        "description": (
            "Fix this customer dataset. "
            "Remove all duplicate rows. Extract pure numbers from the age column (remove text) and replace fully invalid/text strings with the column mean. "
            "Fill missing purchase_amount values with the column mean."
        ),
        "dirty_df": pd.DataFrame(dirty_data),
        "clean_df": pd.DataFrame(clean_data),
        "difficulty": "medium",
        "issues": ["duplicates", "wrong_types", "missing_values", "string_extraction"]
    }

def get_hard_task() -> dict:
    # 15 Rows: Nightmare Data (HTML tags, JSON strings, Datetime chaos, Extreme Outliers)
    dirty_data = {
        "product_id": [1, 2, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13],
        "product_name": ["Apple", "<b>banana</b>", "<b>banana</b>", "CHERRY", "Date\n", "elderberry", "  Fig  ", "GRAPE", "honeydew", "honeydew", 
                         "Kiwi", "Lemon_123", "<i>Mango</i>", "Nectarine", "Orange!"],
        "price": [1.5, 0.5, 0.5, 2.0, 9999.0, 3.0, None, 4.0, 2.5, 2.5, 
                  1.2, -50.0, 3.5, 999.0, 2.8],
        "date_added": ["2024-01-15", "15/02/2024", "15/02/2024", "2024.03.20", "04-01-2024", "May 5th 2024", None, "2024/07/10", "10-08-2024", "10-08-2024",
                       "2024-09-12", "13/10/2024", "Nov 14 2024", "2024.12.15", "2025/01/05"],
        "stock": [100, 200, 200, None, 50, -10, 80, -5, 150, 150, 
                  300, -500, 120, None, -1],
        "category": ["Fruit", "fruit", "fruit", '{"cat": "Fruit"}', "fruit", "Fruit", "fruit", "FRUIT", "Fruit", "Fruit",
                     "fruit", "FRUIT", '{"cat": "fruit"}', "Fruit", "fruit"],
        "rating": [4.5, 3.8, 3.8, 4.2, 1.0, 4.7, None, 4.1, 3.9, 3.9, 
                   9.5, -2.0, 4.8, 50.0, 4.0]
    }
    
    clean_data = {
        "product_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        "product_name": ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon", "Mango", "Nectarine", "Orange"],
        "price": [1.5, 0.5, 2.0, 2.5, 3.0, 2.36, 4.0, 2.5, 1.2, 2.36, 3.5, 2.36, 2.8],
        "date_added": ["2024-01-15", "2024-02-15", "2024-03-20", "2024-01-04", "2024-05-05", None, "2024-07-10", "2024-08-10",
                       "2024-09-12", "2024-10-13", "2024-11-14", "2024-12-15", "2025-01-05"],
        "stock": [100, 200, 125, 50, 0, 80, 0, 150, 300, 0, 120, 125, 0],
        "category": ["Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit", "Fruit"],
        "rating": [4.5, 3.8, 4.2, 4.0, 4.7, 4.1, 4.1, 3.9, 4.1, 4.1, 4.8, 4.1, 4.0]
    }
    
    return {
        "task_id": "hard",
        "description": (
            "Fix this product catalog dataset. "
            "1. Remove duplicates. "
            "2. Strip HTML tags, special characters, and whitespace from product names, then convert to Title Case. "
            "3. Replace outlier prices (<0 or >100) with column mean. "
            "4. Standardize all dates to strict YYYY-MM-DD format regardless of their messy input format. "
            "5. Replace negative stock values with 0, and missing stock with mean. "
            "6. Extract category names from JSON strings if present, then standardize to Title Case. "
            "7. Ensure ratings are clamped between 1.0 and 5.0 (replace invalid ones with mean)."
        ),
        "dirty_df": pd.DataFrame(dirty_data),
        "clean_df": pd.DataFrame(clean_data),
        "difficulty": "hard",
        "issues": ["duplicates", "html_tags", "json_parsing", "extreme_outliers", "inconsistent_formats", "invalid_values"]
    }

def get_all_tasks() -> list:
    """Return all 3 tasks in order of difficulty."""
    return [get_easy_task(), get_medium_task(), get_hard_task()]