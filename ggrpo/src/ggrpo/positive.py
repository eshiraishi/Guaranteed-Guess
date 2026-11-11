"""Online rejection-sampling control: `trl grpo` with one change. Each completion's advantage is its reward (1 if it
passed, 0 if not), with no group baseline and no std scaling, so only passing completions move the policy, and groups
where every completion passes still do. Sampler, loss, clipping, learning rate and steps stay GRPO's. This is PSR in
Zhu et al. (NeurIPS 2025, Eq. 3) and, up to loss normalization, RAFT++ in Xiong et al. (arXiv 2504.11343). Records
rollouts and checkpoints as ggrpo.record does.
usage: python -m ggrpo.positive --config grpo_codenet.yaml [any other `trl grpo` argument]"""
import torch

from ggrpo.record import RecordingGRPOTrainer, run


def get_positive_advantages(rewards_per_func: torch.Tensor) -> torch.Tensor:
    """The advantages for TRL's rewards_per_func (completions x reward functions): one binary reward, used as is."""
    if rewards_per_func.shape[1] != 1:
        raise ValueError(f"needs exactly one reward function, got {rewards_per_func.shape[1]}")
    rewards = rewards_per_func[:, 0]
    if not torch.isin(rewards, torch.tensor([0.0, 1.0], device=rewards.device)).all():
        raise ValueError(f"needs binary rewards, got {sorted(set(rewards.tolist()))}")
    return rewards.clone()


class PositiveGRPOTrainer(RecordingGRPOTrainer):
    def get_advantages(self, output: dict) -> torch.Tensor:
        return get_positive_advantages(self.rewards_per_func)


if __name__ == "__main__":
    run(PositiveGRPOTrainer)
