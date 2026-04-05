---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧹 Data Cleaning Environment

A real-world OpenEnv environment where an AI agent learns to clean messy datasets.

## Overview
The agent receives a dirty dataset and must clean it by fixing:
- Missing values
- Duplicate rows
- Wrong data types
- Outliers
- Inconsistent formats

## Tasks
| Task | Difficulty | Issues |
|------|-----------|--------|
| easy | Easy | Missing values |
| medium | Medium | Missing values + duplicates + wrong types |
| hard | Hard | All of above + outliers + inconsistent formats |

## Action Space
- `task_id`: which task to solve
- `cleaned_data`: CSV string of cleaned dataset

## Observation Space
- `task_id`, `description`, `difficulty`, `issues`, `dirty_data`, `step_count`, `done`

## Reward
- Score between 0.0 and 1.0
- Partial credit for each fix applied correctly

## Setup
pip install -r requirements.txt
python inference.py

## Baseline Scores
| Task | Score |
|------|-------|
| easy | ~0.75 |
| medium | ~0.60 |
| hard | ~0.50 |