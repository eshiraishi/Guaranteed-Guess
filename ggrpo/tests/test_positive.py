import pytest
import torch

from ggrpo.positive import get_positive_advantages


def test_positive_advantages_are_the_rewards_without_a_baseline():
    rewards = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    assert get_positive_advantages(rewards).tolist() == [1.0, 0.0, 1.0, 1.0]


def test_positive_advantages_refuse_several_or_non_binary_rewards():
    with pytest.raises(ValueError, match="exactly one"):
        get_positive_advantages(torch.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="binary"):
        get_positive_advantages(torch.tensor([[0.5], [float("nan")]]))
