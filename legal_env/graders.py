"""
Deterministic Graders for the Legal Case Assistant.

Each grader:
  - Operates purely on keyword matching, regex, and structural validation
  - Returns a score between 0.0 and 1.0
  - Requires NO external APIs and has NO randomness
  - Provides detailed textual feedback
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ───────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lower-case and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _keyword_score(
    text: str,
    keywords: List[str],
    min_count: int = 1,
) -> Tuple[float, List[str]]:
    """
    Compute a keyword match score.

    Returns (score, matched_keywords) where score ∈ [0.0, 1.0].
    If min_count > 1 we require at least that many distinct keyword hits.
    """
    normalized = _normalize(text)
    matched = [kw for kw in keywords if kw.lower() in normalized]
    if min_count <= 1:
        return (1.0 if matched else 0.0), matched
    # Require at least min_count distinct matches
    score = min(len(matched) / min_count, 1.0)
    return score, matched


def _check_field_present(text: str, field_name: str) -> bool:
    """
    Heuristically check whether the agent addressed a required field.

    Looks for the field name (with common formatting variants) in the text.
    """
    normalized = _normalize(text)
    variants = [
        field_name.lower().replace("_", " "),
        field_name.lower().replace("_", ""),
        field_name.lower(),
    ]
    # Also check for markdown-header style: **Field Name**
    for v in variants:
        if v in normalized:
            return True
    return False


# ───────────────────────────────────────────────────────────────────────────
# Core Grader
# ───────────────────────────────────────────────────────────────────────────

def grade_response(
    response: str,
    task: Dict[str, Any],
) -> Tuple[float, float, str]:
    """
    Grade a response against a task's rubric.

    Parameters
    ----------
    response : str
        The agent's full textual response.
    task : dict
        Task definition containing ``grading_rubric`` and
        ``expected_output_fields``.

    Returns
    -------
    structural_score : float  (0-1)
        Fraction of expected fields addressed.
    content_score : float     (0-1)
        Weighted keyword / regex match score.
    feedback : str
        Human-readable grading feedback.
    """
    rubric: Dict[str, Any] = task["grading_rubric"]
    expected_fields: List[str] = task["expected_output_fields"]

    # --- Structural scoring -------------------------------------------------
    fields_found = [f for f in expected_fields if _check_field_present(response, f)]
    structural_score = len(fields_found) / len(expected_fields) if expected_fields else 1.0

    feedback_lines: List[str] = []
    missing = set(expected_fields) - set(fields_found)
    if missing:
        feedback_lines.append(
            f"Missing required fields: {', '.join(sorted(missing))}."
        )
    else:
        feedback_lines.append("All required output fields are present.")

    # --- Content scoring (weighted keyword matching) ------------------------
    weighted_sum = 0.0
    total_weight = 0.0

    for field_name, criteria in rubric.items():
        keywords: List[str] = criteria.get("keywords", [])
        weight: float = criteria.get("weight", 0.25)
        min_count: int = criteria.get("min_count", 1)

        total_weight += weight
        score, matched = _keyword_score(response, keywords, min_count)
        weighted_sum += score * weight

        if score >= 1.0:
            feedback_lines.append(
                f"  ✓ {field_name}: Fully satisfied "
                f"(matched: {', '.join(matched)})."
            )
        elif score > 0:
            feedback_lines.append(
                f"  ~ {field_name}: Partially satisfied ({score:.0%}) "
                f"(matched: {', '.join(matched)}). "
                f"Expected ≥{min_count} from {keywords}."
            )
        else:
            feedback_lines.append(
                f"  ✗ {field_name}: Not satisfied. "
                f"Expected keywords: {keywords}."
            )

    content_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    feedback = "\n".join(feedback_lines)
    return structural_score, content_score, feedback


# ───────────────────────────────────────────────────────────────────────────
# Per-task convenience wrappers
# ───────────────────────────────────────────────────────────────────────────

def grade_classification(response: str, task: Dict[str, Any]) -> Tuple[float, float, str]:
    """Grade Task 1: Legal Issue Classification."""
    return grade_response(response, task)


def grade_risk_detection(response: str, task: Dict[str, Any]) -> Tuple[float, float, str]:
    """Grade Task 2: Contract Risk Detection."""
    return grade_response(response, task)


def grade_clause_optimization(response: str, task: Dict[str, Any]) -> Tuple[float, float, str]:
    """Grade Task 3: Clause Optimization."""
    return grade_response(response, task)
