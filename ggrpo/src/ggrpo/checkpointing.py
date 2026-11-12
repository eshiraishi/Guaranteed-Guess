"""`trl sft` with gradient checkpointing only for long micro-batches: one longer than `gradient_checkpointing_above`
tokens recomputes its layers' activations in the backward pass, a shorter one keeps them. Same gradients
(tests/test_checkpointing.py), less recomputation, more memory.
main is TRL 1.16's sft.main reduced to what sft_full.yaml uses (a local dataset folder, a pinned model revision; no
PEFT, quantization, dataset mixtures or Hub push): any other key fails as unknown.
usage: python -m ggrpo.checkpointing --config sft_full.yaml [any other SFTConfig argument]"""
from dataclasses import dataclass

import trl
from datasets import load_dataset
from transformers.modeling_layers import GradientCheckpointingLayer
from trl import SFTConfig, TrlParser


@dataclass
class SFTArguments:
    """The config keys besides SFTConfig's."""
    model_name_or_path: str
    model_revision: str
    dtype: str
    dataset_name: str                   # a local folder of data files, one per split
    dataset_train_split: str
    dataset_test_split: str
    gradient_checkpointing_above: int   # checkpoint only micro-batches longer than this many tokens


class LongInputCheckpointingSFTTrainer(trl.SFTTrainer):
    """SFTTrainer that checkpoints only micro-batches longer than `gradient_checkpointing_above` tokens."""

    def __init__(self, *args, gradient_checkpointing_above: int, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.args.gradient_checkpointing:
            raise ValueError("needs gradient_checkpointing: true, which gives each layer its checkpoint function")
        self.gradient_checkpointing_above = gradient_checkpointing_above

    def training_step(self, model, inputs, *args, **kwargs):
        checkpoint = inputs["input_ids"].shape[-1] > self.gradient_checkpointing_above
        for module in model.modules():
            if isinstance(module, GradientCheckpointingLayer):
                module.gradient_checkpointing = checkpoint
        return super().training_step(model, inputs, *args, **kwargs)


def main() -> None:
    arguments, training_args = TrlParser((SFTArguments, SFTConfig)).parse_args_and_config()
    training_args.model_init_kwargs = {"revision": arguments.model_revision, "dtype": arguments.dtype}
    dataset = load_dataset(arguments.dataset_name)
    trainer = LongInputCheckpointingSFTTrainer(
        model=arguments.model_name_or_path,
        args=training_args,
        train_dataset=dataset[arguments.dataset_train_split],
        eval_dataset=dataset[arguments.dataset_test_split],
        gradient_checkpointing_above=arguments.gradient_checkpointing_above,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
