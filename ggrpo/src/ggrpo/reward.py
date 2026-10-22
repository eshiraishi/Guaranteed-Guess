"""TRL reward function: 1.0 if the harness passes the completion, else 0.0."""
from collections import Counter
from collections.abc import Callable
from typing import get_args

import httpx
import structlog
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
    results = httpx.post(
        str(settings.server.url),
        json=[{"task_id": task, "completion": completion[0]["content"]}
              for task, completion in zip(task_id, completions, strict=True)],
        timeout=None,
    ).raise_for_status().json()   # 404/503 stop the run
    verdicts = [result["verdict"] for result in results]
    log_extra("verdict", verdicts)
    for verdict in get_args(Verdict):
        log_metric(f"verdict/{verdict}", verdicts.count(verdict) / len(verdicts))
    log.info("rewarded", step=trainer_state.global_step, verdicts=Counter(verdicts))
    return [float(result["passed"]) for result in results]
