import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from transformers import Qwen2Config, Qwen2ForCausalLM   # noqa: E402

from ggrpo.checkpointing import LongInputCheckpointingSFTTrainer   # noqa: E402

LENGTHS = [8, 9, 3, 16]   # around the threshold below, in a fixed order


def make_model() -> Qwen2ForCausalLM:
    """A tiny fp32 Qwen2 with gradient checkpointing enabled, as `gradient_checkpointing: true` leaves it."""
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(config).train()
    model.gradient_checkpointing_enable()
    return model


def accumulate_gradients(model: Qwen2ForCausalLM, above: int, monkeypatch) -> dict[str, torch.Tensor]:
    """The trainer's training_step over LENGTHS, with SFTTrainer's own step reduced to a forward and backward."""
    monkeypatch.setattr(trl.SFTTrainer, "training_step", lambda self, model, inputs: model(**inputs).loss.backward())
    trainer = LongInputCheckpointingSFTTrainer.__new__(LongInputCheckpointingSFTTrainer)   # the step needs no other state
    trainer.gradient_checkpointing_above = above
    generator = torch.Generator().manual_seed(1)
    for length in LENGTHS:
        input_ids = torch.randint(64, (1, length), generator=generator)
        trainer.training_step(model, {"input_ids": input_ids, "labels": input_ids})
    return {name: parameter.grad for name, parameter in model.named_parameters()}


def test_only_micro_batches_over_the_threshold_are_checkpointed(monkeypatch):
    model = make_model()
    checkpointed = []   # the input length of each layer call that goes through the checkpoint function
    for layer in model.model.layers:
        def count(function, hidden_states, *args, checkpoint=layer._gradient_checkpointing_func, **kwargs):
            checkpointed.append(hidden_states.shape[1])
            return checkpoint(function, hidden_states, *args, **kwargs)
        layer._gradient_checkpointing_func = count
    accumulate_gradients(model, above=8, monkeypatch=monkeypatch)
    assert checkpointed == [9, 9, 16, 16]   # 2 layers each; 8 is not over the threshold


def test_gradients_are_bitwise_equal_to_checkpointing_every_micro_batch(monkeypatch):
    every = accumulate_gradients(make_model(), above=0, monkeypatch=monkeypatch)
    long_only = accumulate_gradients(make_model(), above=8, monkeypatch=monkeypatch)
    assert every.keys() == long_only.keys()
    assert all(torch.equal(every[name], long_only[name]) for name in every)
