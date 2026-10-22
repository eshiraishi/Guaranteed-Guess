"""CLI overrides replace single keys and keep the rest of config.toml."""
from pydantic_settings import CliApp

from ggrpo.settings import Settings


def test_cli_override_merges_with_toml():
    settings = CliApp.run(Settings, cli_args=["--generate.mode", "sample", "--generate.beam_search.num_beams", "4"])
    assert settings.generate.mode == "sample"
    assert settings.generate.beam_search.num_beams == 4
    assert settings.generate.beam_search.repetition_penalty == Settings().generate.beam_search.repetition_penalty
    assert settings.generate.model == Settings().generate.model
