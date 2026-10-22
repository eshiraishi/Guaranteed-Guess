"""Settings from config.toml; entry points also accept CLI overrides (--section.key value)."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


class Data(BaseModel):
    mbpp: Path
    humaneval: Path
    humaneval_tests: Path
    step0_predictions: Path
    train_ids: Path | None = None
    train: Path
    eval: Path


class Prompt(BaseModel):
    template: str
    extract: str


class Server(BaseModel):
    host: str
    port: int
    workers: int
    url: HttpUrl


class Harness(BaseModel):
    cc: list[str]
    asm: list[str]
    ldflags: list[str]
    sandbox: Path
    cache: Path
    build_timeout: float
    run_timeout: float
    cpu_seconds: int
    max_file_bytes: int
    stderr_tail: int


class Decoding(BaseModel):
    num_beams: int
    do_sample: bool
    repetition_penalty: float
    temperature: float | None = None   # sampling-only fields: unset when do_sample is false
    top_k: int | None = None
    top_p: float | None = None


class Generate(BaseModel):
    model: str
    revision: str
    adapter: Path | None = None
    attn_implementation: str
    device: str
    max_total_tokens: int
    mode: Literal["beam_search", "sample"]
    beam_search: Decoding
    sample: Decoding
    out: Path
    seed: int


class Score(BaseModel):
    tasks: Path
    predictions: Path
    column: str
    reference: Path
    out: Path


class Paper(BaseModel):
    script: Path
    seed: int
    sample: bool
    out: Path


class Equivalence(BaseModel):
    completions: Path
    attn_implementation: str
    max_tokens: int
    out: Path
    reference: Path


class Log(BaseModel):
    file: Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(toml_file="config.toml")

    data: Data
    prompt: Prompt
    server: Server
    harness: Harness
    generate: Generate
    score: Score
    paper: Paper
    equivalence: Equivalence
    log: Log

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return init_settings, TomlConfigSettingsSource(settings_cls)
