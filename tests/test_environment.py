"""
Test Suite for the Legal Case Assistant OpenEnv Environment.

Covers:
  - Model serialization / validation
  - Task loading
  - Grader determinism and correctness
  - Reward function boundaries
  - Environment lifecycle (reset → step → done)
  - OpenEnv interface compliance
"""

from __future__ import annotations

import json

import pytest

from legal_env.models import (
    LegalAction,
    LegalObservation,
    LegalReward,
    LegalEnvironmentState,
)
from legal_env.tasks import ALL_TASKS, get_task
from legal_env.graders import grade_response
from legal_env.rewards import compute_reward
from legal_env.server.legal_environment import LegalEnvironment


# ═══════════════════════════════════════════════════════════════════════════
# Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestModels:
    """Ensure all Pydantic models serialize/deserialize cleanly."""

    def test_action_serialization(self):
        action = LegalAction(response="This is a test response.")
        data = action.model_dump()
        assert data["response"] == "This is a test response."
        assert json.dumps(data)

    def test_observation_serialization(self):
        obs = LegalObservation(
            done=False,
            reward=0.0,
            task_id="task_1",
            task_type="classification",
            difficulty="easy",
            prompt="Analyze this case.",
            input_text="Some legal text.",
            expected_output_fields=["primary_legal_category"],
        )
        data = obs.model_dump()
        assert data["task_id"] == "task_1"
        assert json.dumps(data)

    def test_reward_bounds(self):
        reward = LegalReward(total=0.5, structural_score=0.0, content_score=0.0, length_penalty=0.0, repetition_penalty=0.0, step_bonus=0.0, detail="")
        assert 0.0 <= reward.total <= 1.0


    def test_state_serialization(self):
        state = LegalEnvironmentState(episode_id="test-123", step_count=0)
        data = state.model_dump()
        assert data["episode_id"] == "test-123"
        assert json.dumps(data)


# ═══════════════════════════════════════════════════════════════════════════
# Task Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTasks:
    """Validate task definitions."""

    def test_all_tasks_loaded(self):
        assert len(ALL_TASKS) == 3

    def test_task_ids_unique(self):
        ids = [t["task_id"] for t in ALL_TASKS]
        assert len(set(ids)) == 3

    def test_get_task(self):
        t = get_task("task_1")
        assert t["task_type"] == "classification"

    def test_get_task_invalid(self):
        with pytest.raises(ValueError):
            get_task("nonexistent")


# ═══════════════════════════════════════════════════════════════════════════
# Grader Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGraders:
    """Verify deterministic grading."""

    def test_perfect_classification(self):
        """A response hitting all keywords should score high."""
        response = (
            "Primary Legal Category: Employment Law\n"
            "Specific Legal Issue: Wrongful Termination and Retaliation\n"
            "Law Type: Civil\n"
            "Jurisdiction: Federal and State (both)\n"
            "This case involves employment law issues related to wrongful "
            "termination and retaliation under both federal and state statutes."
        )
        task = get_task("task_1")
        structural, content, feedback = grade_response(response, task)
        assert structural > 0.5
        assert content > 0.7

    def test_empty_response(self):
        """An empty response should score near zero."""
        task = get_task("task_1")
        structural, content, feedback = grade_response("", task)
        assert structural == 0.0
        assert content == 0.0

    def test_grading_is_deterministic(self):
        """Same input → same output every time."""
        task = get_task("task_2")
        response = "Risk Level: High. The vendor bears unilateral liability."
        results = [grade_response(response, task) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_partial_credit(self):
        """Partial keyword matches should yield partial scores."""
        task = get_task("task_2")
        response = (
            "Risk Level: High\n"
            "The indemnification is one-sided.\n"
            "Mitigation: negotiate a mutual cap."
        )
        structural, content, feedback = grade_response(response, task)
        assert 0.0 < content < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Reward Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRewards:
    """Test reward function properties."""

    def test_reward_bounds(self):
        """Reward must always be in [0, 1]."""
        reward = compute_reward(1.0, 1.0, "x " * 50, [], 1, 3, "ok")
        assert 0.0 <= reward.total <= 1.0

    def test_short_response_penalty(self):
        """Responses under 15 words get a heavy penalty."""
        short = compute_reward(1.0, 1.0, "too short", [], 1, 3, "ok")
        long = compute_reward(1.0, 1.0, "word " * 50, [], 1, 3, "ok")
        assert short.total < long.total
        assert short.length_penalty < 0

    def test_repetition_penalty(self):
        """Repeated identical responses yield score = 0."""
        prev = ["hello world this is a duplicated response"]
        reward = compute_reward(
            1.0, 1.0,
            "hello world this is a duplicated response",
            prev, 2, 3, "ok",
        )
        assert reward.total == 0.0
        assert reward.repetition_penalty < 0


# ═══════════════════════════════════════════════════════════════════════════
# Environment Lifecycle Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvironment:
    """End-to-end environment tests."""

    def test_reset_returns_observation(self):
        env = LegalEnvironment()
        obs = env.reset()
        assert isinstance(obs, LegalObservation)
        assert obs.task_id == "task_1"
        assert obs.difficulty == "easy"

    def test_step_returns_observation(self):
        env = LegalEnvironment()
        env.reset()
        action = LegalAction(
            response=(
                "Primary Legal Category: Employment Law. "
                "Specific Legal Issue: Wrongful Termination. "
                "Law Type: Civil. "
                "Jurisdiction: Federal."
            )
        )
        obs = env.step(action)
        assert isinstance(obs, LegalObservation)
        assert obs.reward > 0.0
        assert isinstance(obs.done, bool)

    def test_state_returns_environment_state(self):
        env = LegalEnvironment()
        env.reset()
        state = env.state
        assert isinstance(state, LegalEnvironmentState)
        assert state.current_task_index == 0
        assert state.done is False

    def test_full_episode_terminates(self):
        """Run through all tasks with weak responses; episode must end."""
        env = LegalEnvironment()
        env.reset()
        for _ in range(20):  # safety limit
            action = LegalAction(
                response="Placeholder legal analysis with sufficient words for the grader to evaluate properly today."
            )
            obs = env.step(action)
            if obs.done:
                break
        assert obs.done, "Episode should end after exhausting all task steps."

    def test_state_json_serializable(self):
        env = LegalEnvironment()
        env.reset()
        state = env.state
        data = state.model_dump()
        serialized = json.dumps(data)
        assert serialized  # no crash

    def test_done_episode_returns_zero_reward(self):
        """Stepping after done should return 0 reward."""
        env = LegalEnvironment()
        env.reset()
        # exhaust all tasks
        for _ in range(20):
            obs = env.step(LegalAction(response="filler " * 30))
            if obs.done:
                break
        # one more step
        obs = env.step(LegalAction(response="late response " * 10))
        assert obs.done is True
        assert obs.reward == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# OpenEnv Interface Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenEnvCompliance:
    """Verify the environment satisfies the OpenEnv contract."""

    def test_has_reset_method(self):
        env = LegalEnvironment()
        assert callable(getattr(env, "reset", None))

    def test_has_step_method(self):
        env = LegalEnvironment()
        assert callable(getattr(env, "step", None))

    def test_has_state_property(self):
        env = LegalEnvironment()
        assert hasattr(env, "state")  # it's a property

    def test_reset_returns_observation(self):
        env = LegalEnvironment()
        obs = env.reset()
        assert isinstance(obs, LegalObservation)

    def test_step_returns_observation(self):
        env = LegalEnvironment()
        env.reset()
        obs = env.step(LegalAction(response="test " * 20))
        assert isinstance(obs, LegalObservation)

    def test_all_models_json_serializable(self):
        env = LegalEnvironment()
        obs = env.reset()
        json.dumps(obs.model_dump())

        obs2 = env.step(LegalAction(response="test " * 20))
        json.dumps(obs2.model_dump())
        json.dumps(env.state.model_dump())
