"""Settings from config.toml; entry points also accept CLI overrides (--section.key value)."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


class Data(BaseModel):
    mbpp: Path
    humaneval: Path
    humaneval_tests: Path
    beam8_cap16k: Path
    train_ids: Path | None = None
    train: Path
    dev: Path
    sft: Path
    sft_exclude: list[str] = []
    dev_fraction: float
    split_seed: int
    eval: Path


class Prompt(BaseModel):
    template: str
    extract: str
    answer: str


class Server(BaseModel):
    host: str
    port: int
    workers: int
    url: HttpUrl
    timeout: float
    outage_seconds: float


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


class Build(BaseModel):
    source: Path
    x86: list[str]
    arm: list[str]
    opt: str
    workers: int
    out: Path


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


class TrackRun(BaseModel):
    name: str
    evals: dict[int, dict[str, Path]]   # train/global_step -> metric -> scored file


class Track(BaseModel):
    entity: str
    project: str
    exclude: list[str]
    export: Path
    runs: dict[str, TrackRun]           # W&B run id -> run


class Latex(BaseModel):
    out: Path
    decontam: Path
    baseline: str                       # run whose step-0 evals the per-problem flips are counted against
    reported: dict[str, tuple[int, int]]   # metric -> (solved, n) as reported in the GG paper
    labels: dict[str, str]              # run id -> name in figures
    trainer_states: dict[str, Path]     # run id -> TRL trainer_state.json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(toml_file="config.toml", env_file=".env")

    wandb_api_key: SecretStr | None = None   # .env only; hosts that don't log to W&B have no .env
    data: Data
    prompt: Prompt
    server: Server
    harness: Harness
    build: Build
    generate: Generate
    score: Score
    paper: Paper
    equivalence: Equivalence
    log: Log
    track: Track
    latex: Latex | None = None               # only the Mac builds the report

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return init_settings, dotenv_settings, TomlConfigSettingsSource(settings_cls)
