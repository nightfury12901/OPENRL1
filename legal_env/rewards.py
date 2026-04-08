"""
Reward Function for the Legal Case Assistant.

Computes a composite reward in [0.0, 1.0] incorporating:
  - Structural score  (required fields present)
  - Content score     (keyword grading)
  - Length penalty     (< 15 words → heavy penalty)
  - Repetition penalty (duplicate response → score = 0)
  - Step bonus         (earlier answers → higher bonus)

All values are deterministic and bounded to [0.0, 1.0].
"""

from __future__ import annotations

from typing import List

from legal_env.models import LegalReward


def compute_reward(
    structural_score: float,
    content_score: float,
    response: str,
    previous_responses: List[str],
    step_number: int,
    max_steps: int,
    feedback: str,
) -> LegalReward:
    """
    Compute the final reward for a single step.

    Parameters
    ----------
    structural_score : float
        Fraction of expected fields present (0-1).
    content_score : float
        Weighted keyword score (0-1).
    response : str
        The agent's raw response text.
    previous_responses : list[str]
        All responses submitted so far in this task (for loop detection).
    step_number : int
        Current step within this task (1-based).
    max_steps : int
        Maximum steps allowed for this task.
    feedback : str
        Grader feedback string.

    Returns
    -------
    LegalReward
        Fully-populated reward object with total ∈ [0.0, 1.0].
    """
    # ── Base score (weighted blend) ─────────────────────────────────────────
    base = 0.5 * structural_score + 0.5 * content_score

    # ── Length penalty ──────────────────────────────────────────────────────
    word_count = len(response.split())
    if word_count < 15:
        length_pen = -0.4   # heavy penalty
    elif word_count < 30:
        length_pen = -0.15  # moderate penalty
    else:
        length_pen = 0.0

    # ── Repetition penalty ──────────────────────────────────────────────────
    normalized_resp = " ".join(response.lower().split())
    repetition_pen = 0.0
    for prev in previous_responses:
        normalized_prev = " ".join(prev.lower().split())
        if normalized_resp == normalized_prev:
            repetition_pen = -1.0  # force score to 0
            break

    # ── Step bonus (earlier = larger bonus) ─────────────────────────────────
    if max_steps > 1:
        step_bonus = 0.1 * (1.0 - (step_number - 1) / (max_steps - 1))
    else:
        step_bonus = 0.1

    # ── Assemble total ──────────────────────────────────────────────────────
    raw_total = base + length_pen + repetition_pen + step_bonus
    total = max(0.01, min(0.99, raw_total))

    detail_parts = [feedback]
    if length_pen < 0:
        detail_parts.append(
            f"Length penalty applied ({word_count} words)."
        )
    if repetition_pen < 0:
        detail_parts.append(
            "Repetition detected — previous identical response found."
        )
    detail_parts.append(
        f"Step bonus: +{step_bonus:.2f} (step {step_number}/{max_steps})."
    )
    detail_parts.append(f"Final score: {total:.3f}")

    return LegalReward(
        total=round(total, 4),
        structural_score=round(structural_score, 4),
        content_score=round(content_score, 4),
        length_penalty=round(length_pen, 4),
        repetition_penalty=round(repetition_pen, 4),
        step_bonus=round(step_bonus, 4),
        detail="\n".join(detail_parts),
    )
