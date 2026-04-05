import pandas as pd
import numpy as np
from io import StringIO

def get_easy_task():
    """
    Easy Task: Fix missing values in a simple employee dataset
    Agent needs to: fill missing age and salary values
    """
    data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, None, 30, None, 28],
        "salary": [50000, 60000, None, 55000, None],
        "department": ["HR", "IT", "IT", "HR", "Finance"]
    }
    df = pd.DataFrame(data)
    
    # What the perfectly cleaned version looks like
    clean_data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 27.5, 30, 27.5, 28],  # filled with mean
        "salary": [50000, 60000, 55000, 55000, 55000],  # filled with mean
        "department": ["HR", "IT", "IT", "HR", "Finance"]
    }
    clean_df = pd.DataFrame(clean_data)
    
    return {
        "task_id": "easy",
        "description": "Fix missing values in this employee dataset by filling them with column means",
        "dirty_df": df,
        "clean_df": clean_df,
        "difficulty": "easy",
        "issues": ["missing_values"]
    }


def get_medium_task():
    """
    Medium Task: Fix missing values + duplicates + wrong data types
    """
    data = {
        "customer_id": [101, 102, 102, 103, 104, 104],  # duplicates
        "name": ["John", "Jane", "Jane", "Bob", "Alice", "Alice"],
        "age": ["25", "30", "30", "abc", "28", "28"],  # wrong type + invalid
        "purchase_amount": [200.5, None, 150.0, 300.0, None, 250.0],  # missing
        "email": ["john@mail.com", "jane@mail.com", "jane@mail.com", 
                  "bob@mail.com", "alice@mail.com", "alice@mail.com"]
    }
    df = pd.DataFrame(data)
    
    clean_data = {
        "customer_id": [101, 102, 103, 104],
        "name": ["John", "Jane", "Bob", "Alice"],
        "age": [25, 30, 25, 28],  # converted to int, invalid replaced with mean
        "purchase_amount": [200.5, 150.0, 300.0, 250.0],  # filled + deduped
        "email": ["john@mail.com", "jane@mail.com", 
                  "bob@mail.com", "alice@mail.com"]
    }
    clean_df = pd.DataFrame(clean_data)
    
    return {
        "task_id": "medium",
        "description": "Fix duplicates, wrong data types, and missing values in this customer dataset",
        "dirty_df": df,
        "clean_df": clean_df,
        "difficulty": "medium",
        "issues": ["duplicates", "wrong_types", "missing_values"]
    }


def get_hard_task():
    """
    Hard Task: Fix everything + inconsistent formats + outliers
    """
    data = {
        "product_id": [1, 2, 2, 3, 4, 5, 6],
        "product_name": ["Apple", "banana", "banana", "CHERRY", 
                         "Date", "elderberry", "Fig"],
        "price": [1.5, 0.5, 0.5, 2.0, 999.0, 3.0, None],  # 999 is outlier
        "date_added": ["2024-01-15", "15/02/2024", "15/02/2024", 
                       "2024-03-20", "2024-04-01", "05-05-2024", None],  # inconsistent formats
        "stock": [100, 200, 200, None, 50, -10, 80],  # negative stock = invalid
        "category": ["Fruit", "fruit", "fruit", "Fruit", 
                     "fruit", "Fruit", "fruit"]  # inconsistent case
    }
    df = pd.DataFrame(data)
    
    clean_data = {
        "product_id": [1, 2, 3, 4, 5, 6],
        "product_name": ["Apple", "Banana", "Cherry", 
                         "Date", "Elderberry", "Fig"],  # title case
        "price": [1.5, 0.5, 2.0, 2.5, 3.0, 2.5],  # outlier replaced with mean
        "date_added": ["2024-01-15", "2024-02-15", "2024-03-20", 
                       "2024-04-01", "2024-05-05", None],  # uniform format
        "stock": [100, 200, None, 50, 0, 80],  # negative replaced
        "category": ["Fruit", "Fruit", "Fruit", 
                     "Fruit", "Fruit", "Fruit"]  # uniform case
    }
    clean_df = pd.DataFrame(clean_data)
    
    return {
        "task_id": "hard",
        "description": "Fix duplicates, outliers, inconsistent formats, invalid values in this product dataset",
        "dirty_df": df,
        "clean_df": clean_df,
        "difficulty": "hard",
        "issues": ["duplicates", "outliers", "inconsistent_formats", 
                   "invalid_values", "missing_values", "inconsistent_case"]
    }


def get_all_tasks():
    return [get_easy_task(), get_medium_task(), get_hard_task()]