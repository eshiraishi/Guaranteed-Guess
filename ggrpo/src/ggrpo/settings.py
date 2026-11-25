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
    exclude: list[str]                  # MBPP-C task ids dropped from train and dev after the split
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
    min_free_gb: int           # 503 below this much free disk
    tasks: list[Path]                   # task files (jsonl) whose task ids the server scores


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


class HubDataset(BaseModel):
    """A Hugging Face dataset pinned to a commit; download() returns its local snapshot in the shared HF cache."""
    repo: str
    revision: str
    files: str   # glob of the files used

    def download(self) -> Path:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(self.repo, repo_type="dataset", revision=self.revision,
                                      allow_patterns=self.files))


class Codeforces(BaseModel):
    """The Codeforces eval set (Enzo, Nov 22): evaluation only, never trained on or redistributed."""
    problems: HubDataset
    submissions: HubDataset
    languages: list[str]       # programmingLanguage values that are C
    solutions: int             # distinct accepted solutions built per problem, in pick order
    seed: str                  # salt of the pick order and the split: never change it once the test split has been used
    split: dict[str, float]    # fraction of problem groups per split
    max_cpu: float             # CPU seconds the reference may use on its slowest test
    dissent: float             # a problem is multi-answer when >= 2 and this share of the finishing solutions disagree
    out: Path                  # ggrpo.codeforces: one row per problem
    built: Path                # ggrpo.codeforces_build (VM): parts of one row per built solution
    chunk: int                 # problems per part
    candidates: Path           # every solution that passed everything in the VM, in task format, for ggrpo.decontam
    decontam: Path             # ggrpo.decontam's pairs for the candidates (--decontam.out)
    decontam_items: Path       # and its items (--decontam.items): proves the run covered every reference
    workers: int               # ggrpo.codeforces_tasks's reference check on the scoring host
    tasks: dict[str, Path]     # ggrpo.codeforces_tasks: task files per split
    eval: Path                 # official tests of the kept problems, read by ggrpo.official
    drops: Path                # one row per problem: kept, or why it was dropped


class Aoj(BaseModel):
    problems: HttpUrl      # Main API problem list
    testcases: HttpUrl     # Data API
    codenet: HubDataset    # accepted C submissions (CodeNet, C subset)
    metadata: HubDataset   # CodeNet problem titles and one sample case
    language: str          # CodeNet original_language of Aizu C submissions
    delay: float           # seconds between requests
    only: list[Path]       # CodeNet task files whose problems to crawl; [] crawls every AOJ problem
    cache: Path            # every response, so a rerun requests nothing it already has
    out: Path


class Codenet(BaseModel):
    submissions: HubDataset    # accepted C submissions (CodeNet, C subset)
    metadata: HubDataset       # statements, with every sample case, and titles
    aoj_language: str          # CodeNet original_language of Aizu C submissions
    min_outputs: int           # distinct sample outputs an RL problem needs
    split: dict[str, float]    # fractions of the RL problems, by problem (approximate: hashed per problem)
    seed: str                  # salt of the sha256 ranking that assigns the split
    problems: Path
    solutions: Path
    built: Path                # directory of parquet parts written by the VM build
    chunk: int                 # solutions per part
    tasks: dict[str, Path]     # RL task file of each split
    cap: dict[str, int]        # tasks per problem in each split


class Decontam(BaseModel):
    n: int                     # canonical tokens per n-gram
    min_instructions: int      # shortest eval function the asm rule checks
    min_rare: int              # fewest rare n-grams such a function needs
    min_exact: int             # shortest function the exact rule requires
    min_rare_c: int            # fewest rare C n-grams an item needs for the c rule
    max_df: float              # n-grams above this share of the pooled samples are boilerplate
    df_sample: int             # training documents sampled per source for that share
    report: float              # lowest score written
    drop_asm: float            # asm score from which a document is left out
    drop_c: float              # c score from which a document is left out
    evals: list[Path]          # eval task files (jsonl with task_id and prompt)
    c_items: list[Path]        # eval task files that carry each item's C in a c column (Codeforces candidates)
    bringup: HubDataset        # GG's BringUp bench
    bringup_c: Path            # its C sources, one folder per program
    gg: HubDataset             # GG's SFT pairs
    stack: list[HubDataset]    # C of GG's Stack documents
    angha_url: str             # GitHub tarball of AnghaBench at a pinned commit
    angha_c: Path              # its C files, written by ggrpo.anghabench
    canary_c: Path             # planted copies' C, written by ggrpo.canary (Mac)
    canary_x86: Path           # their x86, written by ggrpo.canary_build (VM)
    canary_programs: int       # planted programs per type
    canary_seed: int
    canary_recall: dict[str, float]   # lowest recall per kind of copy (the protocol's bar minus 5 points)
    budget: dict[str, float]   # largest share of each source's documents that eval items may drop
    budget_sft: dict[str, float]   # largest share of the SFT documents that eval and RL items may drop together
    fraction: float            # share of training documents scanned: 1.0, or less for a smoke test
    workers: int
    batch: int                 # documents per worker task
    out: Path
    items: Path                # per eval item, which rules could check it


class Sft(BaseModel):
    max_prompt_tokens: int
    max_answer_tokens: int     # answer in GG's answer format
    max_bytes: int             # longest text that can fit those caps
    train: Path
    val: Path


class Rl(BaseModel):
    sources: list[Path]        # task files whose tasks GRPO trains on
    tokenizer: str             # HF model repo of the base model's tokenizer
    tokenizer_revision: str
    max_prompt_tokens: int
    max_answer_tokens: int     # reference answer, in GG's answer format
    out: Path
    references: Path           # SFT folder (train.parquet: prompt, completion) of every kept task's reference, for H2a


class Sample(BaseModel):
    model: Path                # model folder (the census: the fp16 RL start)
    dtype: Literal["float16", "bfloat16"]
    tasks: Path                # task file (jsonl with task_id and prompt)
    seed: str                  # salt of the fixed random order, sha256(seed:task_id)
    samples: int               # completions per task
    temperature: float         # sampling, as grpo_codenet.yaml
    top_p: float
    top_k: int
    max_tokens: int            # completion cap (grpo_codenet.yaml max_completion_length)
    max_model_len: int
    gpu_memory_utilization: float
    chunk: int                 # tasks per generate-and-score round
    score_batch: int           # completions per request to the reward server
    limit: int | None = None   # only the first this many tasks in the order (the fixed validation subset)
    stop_mixed: int | None = None   # stop once this many tasks have 0 < passes < samples (the census)
    shard: str = "0"           # this process's name among the processes sharing out (one host)
    score: bool = True         # False: write the completions unscored and score them later on the Mac (ggrpo.score_samples),
                               # so a GPU never waits on the reward server (Nov 23: the off-campus tunnel outage)
    out: Path


class ScoreSamples(BaseModel):
    samples: Path              # ggrpo.sample rows written with sample.score = false
    url: HttpUrl               # the VM's /score as the Mac reaches it
    batch: int                 # completions per request to the reward server
    out: Path


class Pool(BaseModel):
    census: Path               # ggrpo.sample's rows from the census
    cap: int                   # pool tasks (C-0's step time; a multiple of prompts_per_step)
    prompts_per_step: int
    passes: int
    out: Path                  # GRPO's train file: task_id, prompt
    references: Path           # SFT folder of the pool tasks' references
    offline_rft: Path          # SFT folder of the pool tasks' passing census samples


class Strata(BaseModel):
    train: list[Literal["gg", "codenet", "mbpp"]]   # document sources the scored model trained on
    decontaminated: bool       # true if it trained without the documents ggrpo.decontam drops
    margin: float              # if the high stratum beats the clean one by more, headline the clean subset
    scored: list[Path]         # scored result files of one model
    out: Path


class Fp16(BaseModel):
    model: Path                # folder of the RL starting model
    out: Path                  # its fp16 view for TRL's vLLM


class Reward(BaseModel):
    random_probability: float  # reward_stub's chance of 1.0
    seed: int


class Decoding(BaseModel):
    num_beams: int
    do_sample: bool
    repetition_penalty: float
    temperature: float | None = None   # sampling-only fields: unset when do_sample is false
    top_k: int | None = None
    top_p: float | None = None
    # "offloaded" keeps the KV cache in host RAM: beam 8 at a 32,000-token budget needs ~7 GB of cache and OOMs a
    # 16 GB card on the longest prompts
    cache_implementation: str | None = None


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


class Differential(BaseModel):
    tasks: Path                # the task file whose problems get a suite
    solutions: Path            # CodeNet accepted C solutions, the voters on a candidate input
    out: Path
    mutations: int             # single-token mutations of the sample inputs per problem
    boundaries: int            # single-token boundary substitutions per problem
    voters: int                # solutions that must agree on a kept input
    voter_attempts: int        # solutions compiled while looking for that many voters
    min_kept: int              # fewer kept inputs than this and the problem keeps its sample tests
    max_output_bytes: int
    workers: int


class Recheck(BaseModel):
    scored: Path               # a .scored.jsonl with task_id, passed and the generation column
    column: str                # the column holding the generation ("completion" in a .scored.jsonl)
    suites: Path               # ggrpo.differential output
    out: Path
    workers: int


class Calibration(BaseModel):
    completions: Path
    column: str                # "completion" (already extracted) or "raw"
    suites: Path
    official: Path             # AOJ official tests, keyed by CodeNet problem id
    out: Path
    workers: int


class Official(BaseModel):
    scored: Path               # a file scored on sample tests (rows with task_id, passed and the completion column)
    column: str                # "completion" (already extracted) or "raw"
    official: list[Path]       # official tests: AOJ's (keyed by CodeNet problem id) and Codeforces' (by problem id)
    out: Path
    workers: int


class X86(BaseModel):
    suites: Path               # ggrpo.differential output
    tasks: Path                # the task file whose prompts hold the x86 source
    out: Path
    host: str                  # the gg-ventura VM, the only place Rosetta 2 still runs macOS binaries
    worker: Path               # the VM-side script, copied there on every run


class Log(BaseModel):
    file: Path


class TrackRun(BaseModel):
    name: str
    evals: dict[int, dict[str, Path]]   # train/global_step -> metric -> scored file
    wandb: str | None = None   # W&B run the evals go to, when the training run must not be written to; else the key


class Track(BaseModel):
    entity: str
    project: str
    exclude: list[str]
    export: Path
    report: str | None = None           # the dashboard's url; ggrpo.dashboard rewrites that report instead of a new one
    runs: dict[str, TrackRun]           # W&B run id -> run


class Latex(BaseModel):
    out: Path
    decontam: Path
    baseline: str                       # run whose step-0 evals the per-problem flips are counted against
    reported: dict[str, tuple[int, int]]   # metric -> (solved, n) as reported in the GG paper
    labels: dict[str, str]              # run id -> name in figures
    trainer_states: dict[str, Path]     # run id -> TRL trainer_state.json


class BoardHost(BaseModel):
    ssh: list[str]            # the whole ssh command, so a jump host or a HostKeyAlias stays data, not code
    repo: Path                # that host's checkout; every lane path below is relative to it


class BoardLane(BaseModel):
    name: str
    host: str                 # a key of Board.hosts
    status: Path              # the append-only lane status file
    rows: Path | None = None  # the output the lane grows; None when the lane writes no rows
    rows_extra: Path | None = None   # a second file the same progress goes to: the beam lanes re-generate every task
                                     # that hit CUDA OOM into beam8_oom_<name>.jsonl, so for hours the rows file does
                                     # not grow and the lane read STALLED while it was working (Nov 19). Counted
                                     # together and de-duplicated, since the lane concatenates the two at the end.
    total: int = 0            # expected rows; 0 leaves the percentage and the ETA blank
    unit: str = "row"
    rows_per_unit: int = 1    # 16 completions per census task
    stall_after_minutes: int | None = None   # overrides Board.stall_after_minutes: beam 8 writes a row every ~5 min
                                             # on a hard problem, so the global 10 called a working lane STALLED
    recovering_stall_after_minutes: int | None = None   # used instead of stall_after_minutes once the lane's last
                                             # marker is RECOVER_START: the OOM recovery path keeps the KV cache in
                                             # host RAM, which takes ~4 h for one task, so the 25 above called the
                                             # working lane STALLED for hours (Nov 19).
    rate_window_minutes: int | None = None   # overrides Board.rate_window_minutes: a lane that alternates generating
                                             # a batch with scoring it needs a window over several of those cycles,
                                             # or the rate reads 6.5 or 11.7 for the same true 8.2 (census, Nov 17)


class BoardLocal(BaseModel):
    tart: Path
    vm: str
    helpers: list[str]


class Board(BaseModel):
    hosts: dict[str, BoardHost]
    lanes: list[BoardLane]
    local: BoardLocal
    history: Path
    rate_window_minutes: int
    stall_after_minutes: int
    report_utc_offset_hours: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(toml_file="config.toml", env_file=".env")

    wandb_api_key: SecretStr | None = None   # .env only; hosts that don't log to W&B have no .env
    data: Data
    prompt: Prompt
    server: Server
    harness: Harness
    build: Build
    codeforces: Codeforces
    aoj: Aoj
    codenet: Codenet
    decontam: Decontam
    sft: Sft
    rl: Rl
    sample: Sample
    pool: Pool
    fp16: Fp16
    strata: Strata
    reward: Reward
    generate: Generate
    score: Score
    paper: Paper
    equivalence: Equivalence
    differential: Differential
    recheck: Recheck
    calibration: Calibration
    official: Official
    score_samples: ScoreSamples
    x86: X86
    log: Log
    track: Track
    latex: Latex | None = None               # only the Mac builds the report
    board: Board | None = None               # only the Mac polls the lanes

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return init_settings, dotenv_settings, TomlConfigSettingsSource(settings_cls)
