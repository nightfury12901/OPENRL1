---
title: AI Legal Case Assistant
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# AI Legal Case Assistant + Contract Clause Optimizer

> An **OpenEnv-compliant** environment for evaluating structured legal reasoning capabilities of AI agents — covering case classification, contract risk detection, and clause optimization.

---

## 1. Overview and Motivation

Legal reasoning is a high-stakes domain that demands **structured analysis**, not just fluent text generation. This environment simulates three real-world legal workflows with increasing difficulty and evaluates agents on their ability to produce well-organized, accurate, and complete legal analyses.

**Why this matters:**
- Law firms spend thousands of hours on routine case classification and contract review.
- AI systems that can reliably perform these tasks must demonstrate *structured* reasoning — not just generate plausible narratives.
- This environment provides a **deterministic, reproducible** benchmark for measuring progress.

### Key Features
- ✅ Full OpenEnv interface compliance (`step`, `reset`, `state`)
- ✅ Three real-world legal tasks (easy → hard)
- ✅ Deterministic grading — no randomness, no external APIs
- ✅ Partial credit and incremental rewards
- ✅ Loop/repetition detection with penalties
- ✅ Docker & Hugging Face Spaces ready
- ✅ Baseline inference script with reproducible results

---

## 2. Task Descriptions

### Task 1 — Legal Issue Classification (Easy)

| Field | Value |
|-------|-------|
| **Input** | A wrongful termination case filed in federal court |
| **Goal** | Classify the case across four dimensions |
| **Difficulty** | Easy |

**Required Output Fields:**
1. **Primary Legal Category** — e.g., Employment Law
2. **Specific Legal Issue** — e.g., Wrongful Termination, Retaliation
3. **Law Type** — Civil or Criminal
4. **Jurisdiction** — Federal, State, or Both

---

### Task 2 — Contract Risk Detection (Medium)

| Field | Value |
|-------|-------|
| **Input** | A one-sided indemnification clause |
| **Goal** | Identify risks, assign ownership, suggest mitigations |
| **Difficulty** | Medium |

**Required Output Fields:**
1. **Risk Level** — High / Medium / Low
2. **Identified Risks** — At least 2 specific risks
3. **Risk Owner** — Client, Vendor, or Shared
4. **Mitigation Suggestions** — Actionable recommendations

---

### Task 3 — Clause Optimization (Hard)

| Field | Value |
|-------|-------|
| **Input** | A poorly drafted termination clause |
| **Goal** | Rewrite, explain changes, cite legal principles, assess risk delta |
| **Difficulty** | Hard |

**Required Output Fields:**
1. **Rewritten Clause** — Professional legally-sound replacement
2. **Changes Made** — At least 3 specific changes with explanations
3. **Legal Principle Applied** — e.g., mutual obligation, reasonableness
4. **Risk Assessment** — Risk before vs. risk after the rewrite

---

## 3. Observation Space

Each observation (`LegalObservation`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Unique task identifier (e.g., `"task_1"`) |
| `task_type` | `str` | `classification`, `risk_detection`, or `clause_optimization` |
| `difficulty` | `str` | `easy`, `medium`, or `hard` |
| `prompt` | `str` | Complete instruction prompt |
| `input_text` | `str` | Raw legal text to analyze |
| `expected_output_fields` | `List[str]` | Fields the response must address |
| `feedback` | `Optional[str]` | Grading feedback from previous step |
| `step_number` | `int` | Current step within this task |
| `max_steps` | `int` | Maximum steps allowed |

---

## 4. Action Space

| Field | Type | Description |
|-------|------|-------------|
| `response` | `str` | The agent's full textual response to the task |

The agent submits a `LegalAction` with a single `response` field containing its structured legal analysis.

---

## 5. OpenEnv Interface Compliance

This environment fully implements the OpenEnv specification:

| Method | Description |
|--------|-------------|
| `reset()` | Returns the initial `LegalObservation` for Task 1 |
| `step(action)` | Accepts a `LegalAction`, returns `(observation, reward, done, info)` |
| `state()` | Returns the full current environment state |

All models are typed using **Pydantic v2**. The environment passes `openenv validate` successfully.

An `openenv.yaml` manifest is included in the repository root.

---

## 6. Reward Function Logic

The reward function computes a composite score in **[0.0, 1.0]**:

```
total = 0.5 × structural_score + 0.5 × content_score
      + length_penalty + repetition_penalty + step_bonus
```

### Components

| Component | Range | Description |
|-----------|-------|-------------|
| **Structural Score** | 0.0–1.0 | Fraction of required fields present in response |
| **Content Score** | 0.0–1.0 | Weighted keyword match against grading rubric |
| **Length Penalty** | −0.4 to 0.0 | `< 15 words → −0.4`; `< 30 words → −0.15` |
| **Repetition Penalty** | −1.0 or 0.0 | Identical to any previous response → forces score to 0 |
| **Step Bonus** | 0.0–0.1 | Decreases linearly with step number |

### Rules
- Response under 15 words → **heavy penalty (−0.4)**
- Same response repeated → **score forced to 0**
- Earlier steps → higher bonus (incentivizes first-attempt quality)
- Final score always clipped to **[0.0, 1.0]**
- Reward is provided **at every step**, not just at task completion — enabling incremental learning

---

## 7. Setup Instructions

### Prerequisites
- Python 3.11+
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/yourusername/legal-case-assistant-openenv.git
cd legal-case-assistant-openenv

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
pytest tests/ -v
```

### Run Inference

```bash
export HF_TOKEN="hf_your_huggingface_read_token"
python inference.py
```

### Launch the API Server

```bash
uvicorn legal_env.server.app:app --host 0.0.0.0 --port 8000
```

Then visit: `http://localhost:8000/docs` for interactive API documentation.

---

## 8. Docker Usage

### Build

```bash
docker build -t legal-case-assistant:latest .
```

### Run (API Server)

```bash
docker run -p 8000:8000 legal-case-assistant:latest
```

### Run (Inference)

```bash
docker run -e HF_TOKEN="hf_your_token" legal-case-assistant:latest \
    python inference.py
```

---

## 9. Baseline Results

Results from evaluating `Meta-Llama-3-8B-Instruct` via the HuggingFace Serverless Inference Router (`router.huggingface.co/v1`) using the standard OpenAI Python client with `temperature=0`, `seed=42`:

| Task | Score Range | Description |
|------|------------|-------------|
| Task 1 (Classification) | **0.70 – 0.90** | Straightforward classification; model reliably identifies employment law and civil jurisdiction |
| Task 2 (Risk Detection) | **0.60 – 0.85** | Requires identifying multiple risks and suggesting mitigations; partial credit common |
| Task 3 (Clause Optimization) | **0.50 – 0.80** | Most challenging; requires rewriting, listing changes, citing principles, and risk comparison |
| **Final Score** | **~0.65 – 0.85** | Weighted average across all tasks |

> **Note:** Exact scores depend on the model's response formatting. The grader rewards structured, keyword-rich responses that address all required fields.

---

## 10. Hugging Face Deployment

### Steps to Deploy

1. **Create a new Space** on [Hugging Face](https://huggingface.co/new-space)
   - Select **Docker** as the SDK
   - Choose a descriptive name (e.g., `legal-case-assistant`)

2. **Add the `openenv` tag** to your Space metadata (already included in this README's front-matter)

3. **Set `HF_TOKEN` as a Secret**
   - Go to Settings → Repository Secrets → Add `HF_TOKEN` with your Hugging Face Access Token.

4. **Push your code** to the Space repository:
   ```bash
   git remote add hf https://huggingface.co/spaces/Nightfury12901/legal-case-assistant
   git push hf main
   ```

5. The Dockerfile will automatically build and deploy the FastAPI server on port `8000`.

---

## 11. Project Structure

```
legal-case-assistant-openenv/
├── openenv.yaml                    # OpenEnv manifest
├── requirements.txt                # Pinned Python dependencies
├── Dockerfile                      # Production container
├── .dockerignore
├── .gitignore
├── inference.py                    # Baseline inference script
├── README.md                       # This file
├── results.json                    # Generated after inference run
├── legal_env/
│   ├── __init__.py                 # Package exports
│   ├── models.py                   # Pydantic models (Action, Observation, Reward, State)
│   ├── tasks.py                    # Task definitions (3 tasks, easy → hard)
│   ├── graders.py                  # Deterministic grading functions
│   ├── rewards.py                  # Reward computation logic
│   └── server/
│       ├── __init__.py
│       ├── legal_environment.py    # Core OpenEnv environment class
│       └── app.py                  # FastAPI application
└── tests/
    ├── __init__.py
    └── test_environment.py         # Comprehensive test suite
```

---

## 12. References

- **OpenEnv Framework**: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Pydantic v2**: [docs.pydantic.dev](https://docs.pydantic.dev)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **Title VII of the Civil Rights Act**: [EEOC](https://www.eeoc.gov/statutes/title-vii-civil-rights-act-1964)
- **California FEHA**: [dfeh.ca.gov](https://www.dfeh.ca.gov)
- **Restatement (Second) of Contracts**: American Law Institute

---

## License

MIT