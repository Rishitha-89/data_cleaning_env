---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧹 Data Cleaning Environment — OpenEnv

> Built for the Meta x PyTorch OpenEnv Hackathon 2026 by **Team N+1**

---

## 🌍 Why Data Cleaning?

Data scientists spend **60-80% of their time cleaning data** before any meaningful analysis or model training can begin. Yet there is no standardized way to train or evaluate AI agents on this critical skill.

This environment fills that gap.

We built a reproducible, programmatically graded RL environment where AI agents learn to clean real-world messy datasets — the same kinds of issues every data team faces daily:

- HR systems with incomplete employee records
- Customer databases with duplicate entries and type mismatches  
- Product catalogs with outliers, inconsistent formatting, and invalid values

A well-trained agent on this environment could **automate hours of manual data cleaning work in seconds** — with direct value for data engineering, MLOps, and enterprise analytics pipelines.

---

## 🏗️ Architecture
data-cleaning-env/
├── data_cleaning_env/
│   ├── models.py        # Pydantic typed models (Observation, Action, Reward)
│   ├── tasks.py         # 3 task definitions with dirty + clean datasets
│   ├── graders.py       # Programmatic scoring logic (0.001 - 0.99)
│   └── env.py           # Core environment (reset/step/state/close)
├── server/
│   └── app.py           # FastAPI server exposing OpenEnv endpoints
├── inference.py         # Baseline LLM agent script
├── openenv.yaml         # OpenEnv spec manifest
└── Dockerfile           # Container definition

---

## 🎯 Tasks

### Task 1: Easy — Employee HR Dataset
**Real-world scenario:** HR system exports often have incomplete employee profiles.

| Issue | Fix |
|-------|-----|
| Missing age values | Fill with column mean |
| Missing salary values | Fill with column mean |
| Missing years_exp values | Fill with column mean |

**Passing threshold:** 0.60 / 1.0

---

### Task 2: Medium — Customer Database
**Real-world scenario:** Legacy CRM exports have duplicate records and type mismatches.

| Issue | Fix |
|-------|-----|
| Duplicate customer rows | Remove duplicates |
| Age stored as string with invalid values ("abc") | Convert to numeric, replace with mean |
| Missing purchase amounts | Fill with column mean |

**Passing threshold:** 0.50 / 1.0

---

### Task 3: Hard — Product Catalog
**Real-world scenario:** E-commerce databases merged from multiple sources.

| Issue | Fix |
|-------|-----|
| Duplicate products | Remove duplicates |
| Inconsistent product name casing | Standardize to Title Case |
| Outlier prices (999.0) | Replace with column mean |
| 3 different date formats | Standardize to YYYY-MM-DD |
| Negative stock values | Replace with 0 |
| Inconsistent category casing | Standardize to Title Case |
| Rating outliers | Keep within 1.0–5.0 range |

**Passing threshold:** 0.40 / 1.0

---

## 📐 OpenEnv API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Submit cleaned data, get reward |
| GET | `/state` | Current environment state |
| GET | `/tasks` | List all tasks |

### Observation Space
```json
{
  "task_id": "easy | medium | hard",
  "description": "Natural language task description",
  "difficulty": "easy | medium | hard",
  "issues": ["list", "of", "issues"],
  "dirty_data": "CSV string of messy dataset",
  "step_count": 1,
  "done": false,
  "previous_score": 0.41
}
```

### Action Space
```json
{
  "task_id": "easy | medium | hard",
  "cleaned_data": "CSV string of cleaned dataset"
}
```

### Reward Space
```json
{
  "reward": 0.75,
  "done": false,
  "passed": true,
  "feedback": "Task 'easy' scored 0.750/1.0 — Passed ✅"
}
```

---

## 📊 Reward Function Design

Our reward function provides **rich partial credit signals** at every step rather than sparse binary rewards. This enables agents to learn incrementally from each cleaning action.
Easy Task (max 0.97):
├── No missing values remaining     → 0.29 (partial: 0.14)
├── Age filled correctly            → 0.24 (proportional)
├── Salary filled correctly         → 0.24 (proportional)
├── Years exp filled                → 0.10
└── Other columns unchanged        → 0.10
Medium Task (max 0.97):
├── Duplicates removed              → 0.24 (partial: 0.10)
├── No missing values               → 0.19 (partial: 0.09)
├── Age converted to numeric        → 0.19 (partial: 0.09)
├── Purchase amounts correct        → 0.19 (proportional)
└── No invalid age values           → 0.16 (proportional)
Hard Task (max 0.97):
├── Duplicates removed              → 0.14
├── Product names in Title Case     → 0.14 (proportional)
├── No negative stock               → 0.14 (proportional)
├── Outlier prices replaced         → 0.14
├── Category case consistent        → 0.14
├── Dates in YYYY-MM-DD format      → 0.14 (proportional)
└── Ratings within 1.0–5.0 range   → 0.13 (proportional)

**Anti-gaming mechanism:** If an agent makes no improvement after 3 steps, a penalty is applied to discourage infinite loops and reward hacking.

**Score range:** All scores are strictly within (0.001, 0.99) to ensure meaningful gradient signals throughout training.

---

## 🚀 Quick Start

### Run Locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --reload --port 7860
```

### Run with Docker
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Run Baseline Agent
```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
python inference.py
```

---

## 📈 Baseline Scores

Results from running `meta-llama/Llama-3.3-70B-Instruct` against all tasks:

| Task | Difficulty | Score | Passed |
|------|-----------|-------|--------|
| easy | Easy | ~0.75 | ✅ |
| medium | Medium | ~0.60 | ✅ |
| hard | Hard | ~0.50 | ✅ |
| **Average** | | **~0.62** | |

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | None | Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `meta-llama/Llama-3.3-70B-Instruct` | Model identifier |

---

## ✅ Evaluation Criteria Met

| Criterion | Details |
|-----------|---------|
| ✅ Real-world task | Data cleaning — used in every data pipeline globally |
| ✅ 3+ tasks with graders | Easy, Medium, Hard with programmatic 0.001–0.99 scoring |
| ✅ Meaningful rewards | Proportional partial credit at every step |
| ✅ Anti-gaming | Penalty for no improvement after 3 steps |
| ✅ OpenEnv spec | Full reset/step/state/close API compliance |
| ✅ Dockerfile | Clean containerized deployment |
| ✅ HF Space | Live and running |
| ✅ Baseline script | Reproducible scores with inference.py |

---

## 👥 Team

**Team N+1** — Built for the Meta x PyTorch OpenEnv AI Hackathon 2026

---
