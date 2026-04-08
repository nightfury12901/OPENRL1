"""
Inference Script for the Legal Case Assistant OpenEnv Environment.

Mandatory environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    export HF_TOKEN="hf_..."
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    python inference.py
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


# ── Config from environment ──────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")


# ── Mandatory log helpers ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Truncate action to keep line readable
    action_short = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_short!r} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = LegalEnvironment()

    all_rewards:    List[float] = []
    task_scores:    dict        = {}
    steps_taken:    int         = 0
    success:        bool        = False
    score:          float       = 0.0

    # One [START] per episode
    log_start(task="legal-case-assistant", env="legal_env", model=MODEL_NAME)

    try:
        obs = env.reset()

        while not obs.done:
            task_id = obs.task_id
            if task_id == "done":
                break

            # Get LLM response
            response_text = get_llm_response(client, obs.prompt, obs.feedback)

            # Step the environment
            action = LegalAction(response=response_text)
            obs    = env.step(action)

            steps_taken += 1
            reward = obs.reward if obs.reward is not None else 0.0
            all_rewards.append(reward)

            error = None

            # Mandatory [STEP] log
            log_step(
                step=steps_taken,
                action=response_text,
                reward=reward,
                done=obs.done,
                error=error,
            )

            # Track best score per task
            if obs.metadata.get("advanced"):
                tid   = obs.metadata.get("task_id", task_id)
                best  = obs.metadata.get("best_score", reward)
                task_scores[tid] = best

        # Fill in any tasks not captured via 'advanced'
        state = env.state
        for tid, sc in state.task_scores.items():
            if tid not in task_scores:
                task_scores[tid] = sc

        score   = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Inference error: {exc}", flush=True)

    finally:
        # Mandatory [END] log — always emitted
        log_end(success=success, steps=steps_taken, score=score, rewards=all_rewards)

    # ── Save results.json ────────────────────────────────────────────────
    results = {
        "model":       MODEL_NAME,
        "task_scores": {k: round(v, 4) for k, v in sorted(task_scores.items())},
        "final_score": round(score, 4),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] Results saved to results.json", flush=True)


if __name__ == "__main__":
    main()