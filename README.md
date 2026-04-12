---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧹 Data Cleaning Environment

A real-world OpenEnv environment where AI agents learn to clean messy datasets.

## Tasks
| Task | Difficulty | Issues |
|------|-----------|--------|
| easy | Easy | Missing values |
| medium | Medium | Missing values + duplicates + wrong types |
| hard | Hard | All of above + outliers + inconsistent formats |

## API
- POST `/reset` — start episode
- POST `/step` — submit cleaned data
- GET `/state` — current state
- GET `/tasks` — list all tasks
- GET `/health` — health check

## Setup
pip install -r requirements.txt
uvicorn server.app:app --reload

## Baseline Scores
| Task | Score |
|------|-------|
| easy | ~0.75 |
| medium | ~0.60 |
| hard | ~0.50 |