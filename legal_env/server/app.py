"""
FastAPI Application for the Legal Case Assistant OpenEnv Environment.
"""
from typing import Any, Dict, List

from fastapi import FastAPI
from legal_env.models import LegalAction, LegalObservation, LegalEnvironmentState
from legal_env.server.legal_environment import LegalEnvironment
from legal_env.tasks import ALL_TASKS

env = LegalEnvironment()
app = FastAPI(title="Legal Case Assistant", version="1.0.0")


@app.post("/reset", response_model=LegalObservation)
def reset():
    return env.reset()


@app.post("/step", response_model=LegalObservation)
def step(action: LegalAction):
    return env.step(action)


@app.get("/state", response_model=LegalEnvironmentState)
def state():
    return env.state


@app.get("/tasks", response_model=List[Dict[str, Any]])
def tasks():
    """Return all tasks with their grading rubrics so the validator can enumerate graders."""
    return [
        {
            "task_id": t["task_id"],
            "task_type": t["task_type"],
            "difficulty": t["difficulty"],
            "max_steps": t["max_steps"],
            "expected_output_fields": t["expected_output_fields"],
            "grading_rubric": t["grading_rubric"],
        }
        for t in ALL_TASKS
    ]


@app.get("/health")
def health():
    return {"status": "healthy"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)