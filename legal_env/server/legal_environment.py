"""
Legal Environment implementation.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from legal_env.models import (
    LegalAction,
    LegalObservation,
    LegalEnvironmentState,
    LegalReward,
)
from legal_env.tasks import ALL_TASKS
from legal_env.graders import grade_response
from legal_env.rewards import compute_reward


class LegalEnvironment:
    ADVANCE_THRESHOLD: float = 0.85

    def __init__(self) -> None:
        self._state = LegalEnvironmentState(episode_id=str(uuid.uuid4()))
        self._tasks = list(ALL_TASKS)
        self._task_responses: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LegalObservation:
        self._state = LegalEnvironmentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_index=0,
            total_steps=0,
            task_scores={},
            task_feedbacks={},
            done=False,
            history=[],
            previous_responses=[],
        )
        self._task_responses = []
        return self._make_observation(done=False, reward=0.0)

    def step(
        self,
        action: LegalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LegalObservation:
        if self._state.done:
            return self._make_observation(done=True, reward=0.0, feedback="Episode already finished.")

        task = self._tasks[self._state.current_task_index]

        self._state.step_count += 1
        self._state.total_steps += 1

        structural, content, feedback = grade_response(action.response, task)
        current_task_step = len(self._task_responses) + 1

        reward_info: LegalReward = compute_reward(
            structural_score=structural,
            content_score=content,
            response=action.response,
            previous_responses=self._task_responses,
            step_number=current_task_step,
            max_steps=task["max_steps"],
            feedback=feedback,
        )

        self._task_responses.append(action.response)
        self._state.previous_responses.append(action.response)
        self._state.history.append({
            "task_id": task["task_id"],
            "step": current_task_step,
            "score": reward_info.total,
            "response_preview": action.response[:200],
        })

        prev_best = self._state.task_scores.get(task["task_id"], 0.0)
        self._state.task_scores[task["task_id"]] = max(prev_best, reward_info.total)
        self._state.task_feedbacks[task["task_id"]] = feedback

        advanced = False
        advance = (
            reward_info.total >= self.ADVANCE_THRESHOLD
            or current_task_step >= task["max_steps"]
        )

        if advance:
            advanced = True
            next_idx = self._state.current_task_index + 1
            if next_idx >= len(self._tasks):
                self._state.done = True
            else:
                self._state.current_task_index = next_idx
                self._task_responses = []

        obs = self._make_observation(
            done=self._state.done,
            reward=reward_info.total,
            feedback=reward_info.detail,
            reward_breakdown=reward_info.model_dump(),
        )
        obs.metadata.update({
            "task_id": task["task_id"],
            "advanced": advanced,
            "best_score": self._state.task_scores.get(task["task_id"], 0.0),
        })
        return obs

    @property
    def state(self) -> LegalEnvironmentState:
        return self._state

    def _make_observation(
        self,
        done: bool,
        reward: float,
        feedback: str | None = None,
        reward_breakdown: Optional[dict] = None,
    ) -> LegalObservation:
        if self._state.done:
            return LegalObservation(
                done=True,
                reward=reward,
                task_id="done",
                task_type="none",
                difficulty="none",
                prompt="All tasks completed. Episode finished.",
                input_text="",
                expected_output_fields=[],
                feedback=feedback,
                step_number=len(self._task_responses),
                max_steps=0,
                reward_breakdown=reward_breakdown,
            )

        task = self._tasks[self._state.current_task_index]
        prompt = task["prompt"].replace("{input_text}", task["input_text"])

        return LegalObservation(
            done=done,
            reward=reward,
            task_id=task["task_id"],
            task_type=task["task_type"],
            difficulty=task["difficulty"],
            prompt=prompt,
            input_text=task["input_text"],
            expected_output_fields=task["expected_output_fields"],
            feedback=feedback,
            step_number=len(self._task_responses),
            max_steps=task["max_steps"],
            reward_breakdown=reward_breakdown,
        )
