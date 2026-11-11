import json
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from ggrpo.record import PruneOptimizerCallback, RecordingGRPOTrainer   # noqa: E402


class FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, ids, skip_special_tokens):
        return [" ".join(map(str, row)) for row in ids]


def test_rollouts_record_each_completion(tmp_path):
    trainer = RecordingGRPOTrainer.__new__(RecordingGRPOTrainer)   # write_rollouts needs no model
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer.state = SimpleNamespace(global_step=4)
    trainer.num_generations = 2
    trainer.eos_token_ids = [9]
    trainer._tokenizer = trainer.processing_class = FakeTokenizer()
    trainer.reward_func_names = ["reward_pass"]
    trainer.rewards_per_func = torch.tensor([[1.0], [0.0]])
    trainer._logs = {"extra": {"verdict": deque(["older", "pass", "wrong_output"]), "tests": deque(["", "PP", "PF"])}}
    output = {
        "completion_ids": torch.tensor([[5, 9, 0], [6, 7, 8]]),
        "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "sampling_per_token_logps": torch.tensor([[-0.5, -0.25, -9.0], [-1.0, -1.0, -1.0]]),
        "advantages": torch.tensor([0.5, -0.5]),
    }
    trainer.write_rollouts([{"task_id": "t"}, {"task_id": "t"}], output)
    rows = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").open()]
    assert [(row["update"], row["sample"], row["completion"], row["length"], row["truncated"], row["sampler_logp"],
             row["verdict"], row["tests"], row["rewards"], row["advantage"]) for row in rows] == [
        (5, 0, "5 9", 2, False, -0.75, "pass", "PP", {"reward_pass": 1.0}, 0.5),
        (5, 1, "6 7 8", 3, True, -3.0, "wrong_output", "PF", {"reward_pass": 0.0}, -0.5),
    ]


def test_only_the_newest_checkpoint_keeps_optimizer_state(tmp_path):
    for step in (27, 54):
        for name in ("optimizer.pt", "model.safetensors"):
            (tmp_path / f"checkpoint-{step}").mkdir(exist_ok=True)
            (tmp_path / f"checkpoint-{step}" / name).write_text("x")
    callback, args = PruneOptimizerCallback(), SimpleNamespace(output_dir=str(tmp_path))
    callback.on_save(args, SimpleNamespace(global_step=54), None)
    assert not (tmp_path / "checkpoint-27" / "optimizer.pt").exists()
    assert (tmp_path / "checkpoint-27" / "model.safetensors").exists()
    assert (tmp_path / "checkpoint-54" / "optimizer.pt").exists()
    with pytest.raises(FileNotFoundError, match="optimizer.pt"):
        callback.on_save(args, SimpleNamespace(global_step=81), None)
