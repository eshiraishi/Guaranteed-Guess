import contextlib
import json
from collections import deque
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from transformers import Qwen2Config, Qwen2ForCausalLM   # noqa: E402
from trl.trainer.utils import add_fused_lm_head   # noqa: E402

from ggrpo.record import PruneOptimizerCallback, RecordingGRPOTrainer   # noqa: E402


def test_cutting_padding_keeps_log_probs_entropies_and_gradients():
    """Three rows padded like a GRPO generation batch: prompts left-padded to 5 tokens, completions right-padded to 4.
    Log-probs and entropies match bitwise; gradients to fp32 rounding, since TRL's fused LM head runs in fp32."""
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(config).double()
    add_fused_lm_head(model, outputs=("log_probs", "entropy"))
    trainer = RecordingGRPOTrainer.__new__(RecordingGRPOTrainer)   # the forward needs only the accelerator
    trainer.accelerator = SimpleNamespace(autocast=contextlib.nullcontext, is_main_process=True)
    prompt_mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]])
    completion_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]])
    input_ids = torch.randint(1, 64, (3, 9), generator=torch.Generator().manual_seed(1))
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    results = []
    for forward in (trl.GRPOTrainer._get_per_token_logps_and_entropies,
                    RecordingGRPOTrainer._get_per_token_logps_and_entropies):
        model.zero_grad()
        logps, entropies, _ = forward(trainer, model, input_ids, attention_mask, 4, batch_size=1, compute_entropy=True)
        (logps * completion_mask).sum().backward()
        results.append([logps * completion_mask, entropies * completion_mask,
                        *(parameter.grad.clone() for parameter in model.parameters())])
    (logps_full, entropies_full, *gradients_full), (logps_cut, entropies_cut, *gradients_cut) = results
    assert torch.equal(logps_cut, logps_full) and torch.equal(entropies_cut, entropies_full)
    for full, cut in zip(gradients_full, gradients_cut):
        assert (cut - full).abs().max() <= 1e-6 * full.abs().max()


def test_padding_inside_a_row_is_refused():
    trainer = RecordingGRPOTrainer.__new__(RecordingGRPOTrainer)
    with pytest.raises(ValueError, match="left padding"):
        trainer._get_per_token_logps_and_entropies(None, torch.ones(1, 4, dtype=torch.long),
                                                   torch.tensor([[1, 0, 1, 1]]), 2)


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
