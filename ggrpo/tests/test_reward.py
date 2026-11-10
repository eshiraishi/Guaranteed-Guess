"""The reward call waits out an unreachable server and still stops on HTTP errors. Needs the train extra."""
import httpx
import pytest

pytest.importorskip("transformers")
from transformers import TrainerState  # noqa: E402

from ggrpo import reward  # noqa: E402


def call_reward():
    return reward.reward_pass(
        prompts=[[]], completions=[[{"content": "ret"}]], completion_ids=[[0]], task_id=["mbpp/2"],
        trainer_state=TrainerState(), log_extra=lambda *args: None, log_metric=lambda *args: None,
    )


def respond(status, calls, failures=0):
    def post(url, **kwargs):
        calls.append(url)
        if len(calls) <= failures:
            raise httpx.ConnectError("tunnel down")
        return httpx.Response(status, json=[{"verdict": "pass", "passed": True}], request=httpx.Request("POST", url))
    return post


def test_retries_until_the_server_is_back(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", respond(200, calls, failures=2))
    assert call_reward() == [1.0]
    assert len(calls) == 3


def test_http_error_stops_without_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(httpx, "post", respond(503, calls))
    with pytest.raises(httpx.HTTPStatusError):
        call_reward()
    assert len(calls) == 1


def test_random_reward_shuffles_within_groups_and_is_reproducible(monkeypatch):
    verdicts = [{"verdict": "pass" if i % 4 == 0 else "fail", "passed": i % 4 == 0} for i in range(400)]
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: httpx.Response(200, json=verdicts, request=httpx.Request("POST", url)))
    def call():
        return reward.reward_random(
            prompts=[[]] * 400, completions=[[{"content": f"c{i}"}] for i in range(400)], completion_ids=[[0]] * 400,
            task_id=[f"mbpp/{i // 8}" for i in range(400)], trainer_state=TrainerState(), log_extra=lambda *args: None,
            log_metric=lambda *args: None,
        )
    rewards = call()
    real = [float(verdict["passed"]) for verdict in verdicts]
    assert rewards != real                                                        # the rewards no longer follow the verdicts
    assert all(sum(rewards[i:i + 8]) == sum(real[i:i + 8]) for i in range(0, 400, 8))   # but each group keeps its passes
    assert call() == rewards                                                      # same step and groups, same shuffle


def test_stub_reward_never_calls_the_server(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: pytest.fail("reward_stub must not call the server"))
    rewards = reward.reward_stub(
        prompts=[[]] * 400, completions=[[{"content": f"c{i}"}] for i in range(400)], completion_ids=[[0]] * 400,
        task_id=["mbpp/2"] * 400, trainer_state=TrainerState(), log_extra=lambda *args: None,
        log_metric=lambda *args: None,
    )
    assert set(rewards) == {0.0, 1.0}
