"""
Inference Script for the Legal Case Assistant OpenEnv Environment.

Mandatory environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Emits one [START]/[STEP]*/[END] block PER TASK so the validator counts 3 tasks.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

from openai import OpenAI

from legal_env.models import LegalAction
from legal_env.server.legal_environment import LegalEnvironment
from legal_env.tasks import ALL_TASKS
from legal_env.graders import grade_response
from legal_env.rewards import compute_reward


# ── Config from environment ──────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK    = "legal_env"
MAX_STEPS    = 3
SUCCESS_SCORE_THRESHOLD = 0.1


# ── Mandatory log helpers ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_short = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_short} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────
def get_llm_response(client: OpenAI, prompt: str, feedback: Optional[str]) -> str:
    system_msg = (
        "You are an expert legal analyst. Provide detailed, structured "
        "responses to legal analysis tasks. Always include ALL required "
        "sections clearly labeled with their exact names. Be thorough and specific."
    )
    user_msg = prompt
    if feedback:
        user_msg += f"\n\n--- Previous Feedback ---\n{feedback}\nPlease improve your response based on this feedback."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            seed=42,
            max_tokens=2048,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "Error generating response."


# ── Run a single task as a self-contained episode ────────────────────────────
def run_task(client: OpenAI, task: dict) -> float:
    """
    Runs one task as its own episode:
      - emits [START], one or more [STEP], then [END]
      - returns the final score for this task
    """
    task_id   = task["task_id"]
    task_type = task["task_type"]
    max_steps = task["max_steps"]
    prompt    = task["prompt"].replace("{input_text}", task["input_text"])

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:      List[float] = []
    steps_taken:  int         = 0
    best_score:   float       = 0.01
    success:      bool        = False
    prev_responses: List[str] = []
    feedback: Optional[str]   = None

    try:
        for step in range(1, max_steps + 1):
            response_text = get_llm_response(client, prompt, feedback)
            structural, content, fb = grade_response(response_text, task)
            feedback = fb

            reward_info = compute_reward(
                structural_score=structural,
                content_score=content,
                response=response_text,
                previous_responses=prev_responses,
                step_number=step,
                max_steps=max_steps,
                feedback=fb,
            )
            prev_responses.append(response_text)

            reward      = reward_info.total
            done        = step == max_steps or reward >= 0.85
            steps_taken = step
            rewards.append(reward)
            best_score  = max(best_score, reward)

            log_step(step=step, action=response_text, reward=reward, done=done, error=None)

            if done:
                break

        score   = max(0.01, min(0.99, best_score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        score   = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    task_scores: dict = {}

    for task in ALL_TASKS:
        score = run_task(client, task)
        task_scores[task["task_id"]] = round(score, 4)

    final_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    final_score = max(0.01, min(0.99, final_score))

    # ── Save results.json ────────────────────────────────────────────────
    results = {
        "model":       MODEL_NAME,
        "task_scores": {k: round(v, 4) for k, v in sorted(task_scores.items())},
        "final_score": round(final_score, 4),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Results saved to results.json — final_score={final_score:.3f}", flush=True)


if __name__ == "__main__":
    main()