"""
Task Definitions for the Legal Case Assistant.

Each task is a dictionary containing:
- task_id:                Unique identifier
- task_type:              classification | risk_detection | clause_optimization
- difficulty:             easy | medium | hard
- input_text:             The raw legal text supplied to the agent
- prompt:                 Full instruction prompt
- expected_output_fields: Fields the agent's response MUST address
- grading_rubric:         Deterministic criteria used by the grader
- max_steps:              Maximum interaction steps for this task
"""

from __future__ import annotations
from typing import Any, Dict, List

# ───────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Legal Issue Classification
# ───────────────────────────────────────────────────────────────────────────
TASK_1: Dict[str, Any] = {
    "task_id": "task_1",
    "task_type": "classification",
    "difficulty": "easy",
    "max_steps": 3,
    "input_text": (
        "In March 2024, Jane Doe filed a lawsuit against TechCorp Inc. in the "
        "United States District Court for the Northern District of California. "
        "Jane alleges that TechCorp wrongfully terminated her employment in "
        "retaliation for reporting safety violations at the company's San Jose "
        "manufacturing facility. She claims that TechCorp violated Title VII of "
        "the Civil Rights Act and California's Fair Employment and Housing Act "
        "(FEHA). Jane is seeking reinstatement, back pay, compensatory damages "
        "for emotional distress, and punitive damages. TechCorp argues the "
        "termination was due to a company-wide restructuring and was unrelated "
        "to Jane's complaints. The case involves witness testimony from five "
        "former coworkers who corroborate Jane's account of filing safety "
        "complaints with management."
    ),
    "prompt": (
        "Analyze the following case description and provide a structured legal "
        "classification. Your response MUST include ALL of the following clearly "
        "labeled sections:\n\n"
        "1. **Primary Legal Category**: The main area of law (e.g., Employment Law, "
        "Contract Law, Tort Law, etc.)\n"
        "2. **Specific Legal Issue**: The precise legal issue at stake (e.g., "
        "Wrongful Termination, Retaliation, Breach of Contract)\n"
        "3. **Law Type**: Whether this is a 'civil' or 'criminal' matter\n"
        "4. **Jurisdiction**: Whether this falls under 'federal', 'state', or "
        "'both' jurisdiction\n\n"
        "Provide reasoning for each classification.\n\n"
        "Case Description:\n{input_text}"
    ),
    "expected_output_fields": [
        "primary_legal_category",
        "specific_legal_issue",
        "law_type",
        "jurisdiction",
    ],
    "grading_rubric": {
        "primary_legal_category": {
            "keywords": ["employment law", "employment", "labor law", "labor"],
            "weight": 0.25,
        },
        "specific_legal_issue": {
            "keywords": [
                "wrongful termination",
                "retaliation",
                "retaliatory discharge",
                "whistleblower",
            ],
            "weight": 0.25,
        },
        "law_type": {
            "keywords": ["civil"],
            "weight": 0.25,
        },
        "jurisdiction": {
            "keywords": ["federal", "both", "federal and state"],
            "weight": 0.25,
        },
    },
}

# ───────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: Contract Risk Detection
# ───────────────────────────────────────────────────────────────────────────
TASK_2: Dict[str, Any] = {
    "task_id": "task_2",
    "task_type": "risk_detection",
    "difficulty": "medium",
    "max_steps": 3,
    "input_text": (
        "INDEMNIFICATION CLAUSE: The Vendor shall indemnify, defend, and hold "
        "harmless the Client and its officers, directors, employees, and agents "
        "from and against any and all claims, damages, losses, costs, and "
        "expenses (including reasonable attorneys' fees) arising out of or "
        "related to: (a) any breach of the Vendor's representations or "
        "warranties; (b) any negligent or wrongful act or omission of the "
        "Vendor; (c) any violation of applicable laws by the Vendor. The "
        "Vendor's total liability under this Agreement shall not exceed the "
        "total fees paid by the Client in the twelve (12) months preceding the "
        "event giving rise to the claim. This indemnification obligation shall "
        "survive termination of this Agreement for a period of twenty-four (24) "
        "months. The Client shall have no obligation to indemnify the Vendor "
        "under any circumstances. The Vendor waives any right to consequential, "
        "incidental, or punitive damages against the Client."
    ),
    "prompt": (
        "Review the following contract clause and perform a comprehensive risk "
        "analysis. Your response MUST include ALL of the following clearly "
        "labeled sections:\n\n"
        "1. **Risk Level**: Overall risk level ('high', 'medium', or 'low')\n"
        "2. **Identified Risks**: List at least 2 specific risks (label each)\n"
        "3. **Risk Owner**: Who bears the risk — 'client', 'vendor', or 'shared'\n"
        "4. **Mitigation Suggestions**: Provide specific recommendations to "
        "reduce or eliminate each identified risk\n\n"
        "Be thorough and reference specific language from the clause.\n\n"
        "Contract Clause:\n{input_text}"
    ),
    "expected_output_fields": [
        "risk_level",
        "identified_risks",
        "risk_owner",
        "mitigation_suggestions",
    ],
    "grading_rubric": {
        "risk_level": {
            "keywords": ["high"],
            "weight": 0.15,
        },
        "identified_risks": {
            "keywords": [
                "one-sided",
                "unilateral",
                "unlimited",
                "no cap",
                "no mutual",
                "asymmetric",
                "liability cap",
                "liability limit",
                "consequential damages",
                "waiver",
                "indemnif",
                "survive",
                "survival",
            ],
            "min_count": 2,
            "weight": 0.30,
        },
        "risk_owner": {
            "keywords": ["vendor"],
            "weight": 0.15,
        },
        "mitigation_suggestions": {
            "keywords": [
                "mutual",
                "cap",
                "limit",
                "negotiate",
                "insurance",
                "reciprocal",
                "balanced",
                "reduce",
            ],
            "min_count": 2,
            "weight": 0.40,
        },
    },
}

# ───────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Clause Optimization
# ───────────────────────────────────────────────────────────────────────────
TASK_3: Dict[str, Any] = {
    "task_id": "task_3",
    "task_type": "clause_optimization",
    "difficulty": "hard",
    "max_steps": 3,
    "input_text": (
        "TERMINATION: Either party can end this agreement whenever they want "
        "for any reason or no reason at all by telling the other party. Once "
        "terminated, neither party owes anything to the other and all "
        "obligations are immediately void. The company can also terminate if "
        "they think the contractor is not doing good work, and the contractor "
        "cannot dispute this. Any work done by the contractor becomes the "
        "company's property immediately and the contractor gives up all rights "
        "forever. The contractor also cannot work for any competitor for 5 "
        "years after termination anywhere in the world."
    ),
    "prompt": (
        "The following termination clause is poorly written and contains "
        "significant legal issues. Rewrite it as a professional, legally sound "
        "clause and provide a full analysis.\n\n"
        "Your response MUST include ALL of the following clearly labeled "
        "sections:\n\n"
        "1. **Rewritten Clause**: A professionally drafted replacement clause\n"
        "2. **Changes Made**: List at least 3 specific changes with explanations\n"
        "3. **Legal Principle Applied**: Name the legal principle(s) that "
        "guided your rewrite (e.g., mutual obligation, reasonableness, "
        "consideration)\n"
        "4. **Risk Assessment**: Describe risk BEFORE the rewrite and risk "
        "AFTER the rewrite\n\n"
        "Be specific and reference the original language you changed.\n\n"
        "Original Clause:\n{input_text}"
    ),
    "expected_output_fields": [
        "rewritten_clause",
        "changes_made",
        "legal_principle",
        "risk_assessment",
    ],
    "grading_rubric": {
        "rewritten_clause": {
            "keywords": [
                "notice",
                "days",
                "written",
                "material breach",
                "cure",
                "obligations",
            ],
            "min_count": 3,
            "weight": 0.30,
        },
        "changes_made": {
            "keywords": [
                "notice period",
                "non-compete",
                "intellectual property",
                "dispute",
                "reasonable",
                "mutual",
                "cure period",
                "scope",
                "geographic",
                "duration",
            ],
            "min_count": 3,
            "weight": 0.25,
        },
        "legal_principle": {
            "keywords": [
                "mutual",
                "reasonable",
                "good faith",
                "consideration",
                "enforceab",
                "proportional",
                "fair dealing",
                "unconscionab",
            ],
            "weight": 0.20,
        },
        "risk_assessment": {
            "keywords": [
                "before",
                "after",
                "high",
                "low",
                "medium",
                "reduced",
                "mitigated",
                "improved",
            ],
            "min_count": 2,
            "weight": 0.25,
        },
    },
}


# ───────────────────────────────────────────────────────────────────────────
# Ordered list of all tasks (easy → hard)
# ───────────────────────────────────────────────────────────────────────────
ALL_TASKS: List[Dict[str, Any]] = [TASK_1, TASK_2, TASK_3]


def get_task(task_id: str) -> Dict[str, Any]:
    """Look up a task by its ID. Raises ValueError if not found."""
    for task in ALL_TASKS:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown task_id: {task_id}")
