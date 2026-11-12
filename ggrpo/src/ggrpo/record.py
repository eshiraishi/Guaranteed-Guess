"""`trl grpo` that records what the paper needs after the runs (prereg.md §2, Recording):
- every rollout, one line per completion in <output_dir>/rollouts.jsonl: the update it trains (1-based), task_id, its
  position in the group, the completion text, its length in tokens (EOS included), whether it was truncated, the
  sampler's log-probability summed over the completion, the reward server's verdict and per-test outcomes (P/F per stdio
  test; empty for C-driver tasks), each reward function's reward, and the advantage trained on. A resumed run
  regenerates the updates after its checkpoint, so an update can appear twice: its last lines are the ones trained on;
- optimizer state only in the newest checkpoint: older checkpoints keep weights, scheduler and RNG state.
usage: python -m ggrpo.record --config grpo_codenet.yaml [any other `trl grpo` argument]
GRPO as is; the permuted arm adds --reward_funcs ggrpo.reward.reward_random; online RFT is ggrpo.positive, which
records the same way."""
import json
from pathlib import Path

import torch
import trl
from transformers import TrainerCallback, set_seed
from trl.scripts import grpo


class PruneOptimizerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        newest = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not (newest / "optimizer.pt").exists():
            raise FileNotFoundError(f"{newest} has no optimizer.pt: a run could not resume from it (save_only_model?)")
        for checkpoint in Path(args.output_dir).glob("checkpoint-*"):
            if checkpoint != newest:
                (checkpoint / "optimizer.pt").unlink(missing_ok=True)


class RecordingGRPOTrainer(trl.GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(PruneOptimizerCallback())
        # colocated vLLM samples from the global CUDA RNG and reset it to seed 0 at warm-up; the run's seed must set it
        set_seed(self.args.seed)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
        """TRL's log-prob forward, one row at a time without its padding: every row is as wide as its generation batch
        (longest prompt + longest completion), so this gives the same values on fewer tokens."""
        if kwargs.get("compute_aux_loss") or any(value is not None for key, value in kwargs.items()
                                                 if key not in ("batch_size", "compute_entropy", "compute_aux_loss")):
            raise ValueError(f"written for text models without an aux loss: {kwargs}")
        logps, entropies = [], []
        for ids, mask in zip(input_ids, attention_mask):
            kept = mask.nonzero().flatten()
            start, end = kept[0].item(), kept[-1].item() + 1   # prompts are left-padded, completions right-padded
            right_pad = input_ids.size(1) - end
            if not mask[start:end].all() or right_pad >= logits_to_keep:
                raise ValueError(f"a row is not left padding, prompt, completion, right padding: {mask.tolist()}")
            row_logps, row_entropies, _ = super()._get_per_token_logps_and_entropies(
                model, ids[None, start:end], mask[None, start:end], logits_to_keep - right_pad, **kwargs)
            logps.append(torch.nn.functional.pad(row_logps, (0, right_pad)))
            if row_entropies is not None:
                entropies.append(torch.nn.functional.pad(row_entropies, (0, right_pad)))
        return torch.cat(logps), torch.cat(entropies) if entropies else None, None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        self.rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        return self.rewards_per_func

    def get_advantages(self, output: dict) -> torch.Tensor:
        """The advantages trained on: GRPO's own; ggrpo.positive replaces them."""
        return output["advantages"]

    def _generate_and_score_completions(self, inputs):
        if self.accelerator.num_processes != 1:
            raise ValueError("written for one process: rewards_per_func and the logs would need the process slice")
        output = super()._generate_and_score_completions(inputs)
        output["advantages"] = self.get_advantages(output)
        self.write_rollouts(inputs, output)
        return output

    def write_rollouts(self, inputs: list[dict], output: dict) -> None:
        lengths = output["completion_mask"].sum(dim=1).tolist()
        ids = [row[:length].tolist() for row, length in zip(output["completion_ids"], lengths, strict=True)]
        # no IS correction, no sampler log-probs: fail here rather than record a run without them
        logps = (output["sampling_per_token_logps"] * output["completion_mask"]).sum(dim=1).tolist()
        ends = {*self.eos_token_ids, self._tokenizer.pad_token_id}
        extra = {column: list(values)[-len(ids):] for column, values in self._logs["extra"].items()}
        rows = [
            {
                "update": self.state.global_step + 1,
                "task_id": inputs[position]["task_id"],
                "sample": position % self.num_generations,
                "completion": text,
                "length": lengths[position],
                "truncated": ids[position][-1] not in ends,
                "sampler_logp": logps[position],
                **{column: values[position] for column, values in extra.items()},
                "rewards": dict(zip(self.reward_func_names, self.rewards_per_func[position].tolist(), strict=True)),
                "advantage": output["advantages"][position].item(),
            }
            for position, text in enumerate(self.processing_class.batch_decode(ids, skip_special_tokens=True))
        ]
        with (Path(self.args.output_dir) / "rollouts.jsonl").open("a") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)


def run(trainer_class: type[trl.GRPOTrainer]) -> None:
    trl.GRPOTrainer = trainer_class   # grpo.main imports GRPOTrainer from trl when it runs
    parser = grpo.make_parser()
    grpo.main(*parser.parse_args_and_config(fail_with_unknown_args=False))


if __name__ == "__main__":
    run(RecordingGRPOTrainer)
