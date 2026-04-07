---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧹 Data Cleaning Environment

A real-world OpenEnv environment built for the Meta x Scaler Hackathon where an AI agent learns to clean messy datasets.

## Overview
Data cleaning is an essential but tedious part of ML pipelines. In this environment, the agent receives a dirty dataset and must clean it by fixing:
- Missing values (imputation)
- Duplicate rows
- Wrong data types
- Outliers (e.g., negative stock values)
- Inconsistent formats (e.g., datetime parsing, casing)

## Tasks
| Task | Difficulty | Issues |
|------|-----------|--------|
| `easy` | Easy | Missing values |
| `medium` | Medium | Missing values + duplicates + wrong types |
| `hard` | Hard | All of above + outliers + inconsistent formats + invalid values |

## 🧠 Advanced Reward Shaping
Unlike basic binary pass/fail environments, this environment utilizes **granular reward shaping**. The grading script evaluates the AI's submission column-by-column, awarding partial credit for partial fixes. It also penalizes the agent for getting stuck in infinite loops without improving.

## Environment Details
- **Action Space:** `task_id` (which task to solve) & `cleaned_data` (CSV string of cleaned dataset)
- **Observation Space:** `task_id`, `description`, `difficulty`, `issues`, `dirty_data`, `step_count`, `done`
- **Reward:** Score between 0.0 and 1.0 (with partial credit)

## Setup & Inference
```bash
pip install -r requirements.txt
python inference.py