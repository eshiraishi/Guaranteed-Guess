"""TRL reward function: 1.0 if the harness passes the completion, else 0.0."""
from collections import Counter
from collections.abc import Callable
from typing import get_args

import httpx
import structlog
from tenacity import Retrying, retry_if_exception_type, stop_after_delay, wait_exponential
from transformers import TrainerState

from ggrpo.harness import Verdict
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")


def reward_pass(
    prompts: list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    completion_ids: list[list[int]],
    task_id: list[str],
    trainer_state: TrainerState,
    log_extra: Callable[[str, list], None],
    log_metric: Callable[[str, float], None],
) -> list[float]:
    """The parameters are exactly what trl's GRPOTrainer passes; task_id is the only extra dataset column."""
    items = [{"task_id": task, "completion": completion[0]["content"]}
             for task, completion in zip(task_id, completions, strict=True)]
    # Wait out an unreachable server (the SSH tunnel to the Mac VM) instead of crashing; scoring is a pure function of
    # the items, so resending is safe. HTTP errors (404/503) are not transport errors and still stop the run.
    for attempt in Retrying(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(max=60),
        stop=stop_after_delay(settings.server.outage_seconds),
        before_sleep=lambda state: log.warning("reward server unreachable", step=trainer_state.global_step,
                                               attempt=state.attempt_number, error=repr(state.outcome.exception())),
        reraise=True,
    ):
        with attempt:
            results = httpx.post(
                str(settings.server.url), json=items, timeout=settings.server.timeout
            ).raise_for_status().json()
    verdicts = [result["verdict"] for result in results]
    log_extra("verdict", verdicts)
    for verdict in get_args(Verdict):
        log_metric(f"verdict/{verdict}", verdicts.count(verdict) / len(verdicts))
    log.info("rewarded", step=trainer_state.global_step, verdicts=Counter(verdicts))
    return [float(result["passed"]) for result in results]
