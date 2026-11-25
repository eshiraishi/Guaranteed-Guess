"""Build each CodeNet problem's differential test suite (prereg.md §4, Differential tests): mutate the sample inputs,
keep an input when the reference ARM answers it the same way on two runs and the problem's accepted C solutions all
print that answer, and record the agreed answer as the expected output.

The x86 source is the third voter in the prereg; it needs Rosetta 2, so it runs only where `include_x86` is set. A
problem is usable when it keeps at least `min_kept` inputs with at least 2 distinct outputs; otherwise the sample
tests stand, as the prereg says."""
import json
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Iterator
from pathlib import Path

import polars as pl
from pydantic import BaseModel
from pydantic_settings import CliApp

from ggrpo.harness import execute, link, normalize, run
from ggrpo.settings import Settings

settings = Settings()   # the TOML values, which a spawned pool worker re-reads on import; main() parses the CLI
piece = re.compile(r"-?\d+\.\d+|-?\d+|[A-Za-z]+")
numeric = re.compile(r"-?\d+(?:\.\d+)?$")
BOUNDARIES = (0, 1, -1, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63))
OPERATIONS = ("negate", "increment", "decrement", "double", "halve")


class Suite(BaseModel):
    """One problem's generated tests and the counts behind them."""
    task_id: str
    problem_id: str
    tests: list[dict[str, str]]          # the sample tests first, then the kept generated ones
    samples: int
    candidates: int
    kept: int
    distinct_outputs: int
    voters: int
    rejected: dict[str, int]

    @property
    def usable(self) -> bool:
        return self.kept >= settings.differential.min_kept and self.distinct_outputs >= 2


def apply_operation(token: str, operation: str) -> str:
    """One numeric token under one fixed mutation; floats keep their shape."""
    if "." in token:
        value = float(token)
        changed = {"negate": -value, "increment": value + 1.0, "decrement": value - 1.0,
                   "double": value * 2.0, "halve": value / 2.0}[operation]
        return f"{changed:.6g}"
    value = int(token)
    changed = {"negate": -value, "increment": value + 1, "decrement": value - 1,
               "double": value * 2, "halve": value // 2}[operation]
    return str(changed)


def substitute(text: str, replacements: dict[int, str]) -> str:
    """The text with the given numeric tokens replaced, applied right to left so the spans stay valid."""
    spans = [match.span() for match in piece.finditer(text)]
    for index in sorted(replacements, reverse=True):
        start, end = spans[index]
        text = text[:start] + replacements[index] + text[end:]
    return text


def rearrange(word: str) -> list[str]:
    """Length- and alphabet-preserving rearrangements of a word token, so the problem's constraints still hold."""
    variants = [word[1:] + word[0], word[::-1], word[-1] + word[1:], word[:-1] + word[0]]
    if len(word) >= 2:
        variants.append(word[1] + word[0] + word[2:])
    return variants


def mutation_values(token: str) -> list[str]:
    if numeric.match(token):
        return [apply_operation(token, operation) for operation in OPERATIONS]
    return rearrange(token)


def boundary_values(token: str) -> list[str]:
    if numeric.match(token):
        return [str(value) for value in BOUNDARIES]
    return [token[0] * len(token), token[-1] * len(token), "".join(sorted(token)),
            "".join(sorted(token, reverse=True))]


def enumerate_variants(text: str, values: Callable[[str], list[str]]) -> Iterator[str]:
    """Every single-token substitution, then every two-token one, in a fixed order."""
    tokens = [match.group() for match in piece.finditer(text)]
    for index, token in enumerate(tokens):
        for value in values(token):
            yield substitute(text, {index: value})
    for first in range(len(tokens)):
        for second in range(first + 1, len(tokens)):
            for left in values(tokens[first]):
                for right in values(tokens[second]):
                    yield substitute(text, {first: left, second: right})


def take_round_robin(streams: list[Iterator[str]], quota: int) -> list[str]:
    """Up to quota variants, one per sample in turn, so no single sample fills the list."""
    taken: list[str] = []
    active = list(streams)
    while active and len(taken) < quota:
        for stream in list(active):
            variant = next(stream, None)
            if variant is None:
                active.remove(stream)
                continue
            taken.append(variant)
            if len(taken) >= quota:
                break
    return taken


def build_candidates(samples: list[dict[str, str]]) -> list[tuple[str, str]]:
    """The problem's candidate inputs: mutations of the sample inputs, then boundary substitutions."""
    taken = [(text, "mutation") for text in take_round_robin(
        [enumerate_variants(sample["input"], mutation_values) for sample in samples],
        settings.differential.mutations)]
    taken += [(text, "boundary") for text in take_round_robin(
        [enumerate_variants(sample["input"], boundary_values) for sample in samples],
        settings.differential.boundaries)]
    originals = {sample["input"] for sample in samples}
    seen, unique = set(originals), []
    for text, origin in taken:
        if text not in seen:
            seen.add(text)
            unique.append((text, origin))
    return unique


def read_output(work: Path, program: str, stdin: str) -> str | None:
    """The program's normalized stdout, or None when it timed out, hit a signal or printed too much."""
    finished = execute(work, stdin + "\n", program)
    if finished is None or finished.returncode < 0:
        return None
    if len(finished.stdout) > settings.differential.max_output_bytes:
        return None
    return normalize(finished.stdout)


def build_voters(work: Path, sources: list[str], samples: list[dict[str, str]]) -> list[str]:
    """Compile the problem's C solutions and keep the ones that pass its sample tests: the voters on an input."""
    voters = []
    for position, source in enumerate(sources[:settings.differential.voter_attempts]):
        name = f"sol{position}"
        (work / f"{name}.c").write_text(source)
        built = run([*settings.harness.cc, f"{name}.c", "-o", name, *settings.harness.ldflags],
                    cwd=work, timeout=settings.harness.build_timeout)
        if built.returncode:
            continue
        if all(read_output(work, f"./{name}", s["input"]) == normalize(s["output"]) for s in samples):
            voters.append(f"./{name}")
        if len(voters) == settings.differential.voters:
            break
    return voters


def build_suite(task_id: str, reference: str, samples: list[dict[str, str]], sources: list[str]) -> Suite:
    """One problem's suite. The reference must build and answer every sample test, as the prereg requires."""
    problem_id = task_id.split("/")[1]
    rejected: Counter = Counter()
    kept: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="gg-diff-") as tmp:
        work = Path(tmp).resolve()
        if failure := link(work, reference, [], 0.0):
            raise RuntimeError(f"{task_id}: the reference does not build ({failure.verdict}): {failure.stderr}")
        voters = build_voters(work, sources, samples)
        if not voters:
            rejected["no_voter_solution"] += 1
            return Suite(task_id=task_id, problem_id=problem_id, tests=samples, samples=len(samples),
                         candidates=0, kept=0, distinct_outputs=0, voters=0, rejected=dict(rejected))
        candidates = build_candidates(samples)
        for text, origin in candidates:
            answer = read_output(work, "./a.out", text)
            if answer is None:
                rejected[f"reference_unusable_{origin}"] += 1
                continue
            if read_output(work, "./a.out", text) != answer:
                rejected["reference_nondeterministic"] += 1
                continue
            if not answer:
                rejected["empty_output"] += 1
                continue
            if any(read_output(work, voter, text) != answer for voter in voters):
                rejected[f"solutions_disagree_{origin}"] += 1
                continue
            kept.append({"input": text, "output": answer})
    outputs = {test["output"] for test in kept} | {normalize(s["output"]) for s in samples}
    return Suite(task_id=task_id, problem_id=problem_id, tests=samples + kept, samples=len(samples),
                 candidates=len(candidates), kept=len(kept), distinct_outputs=len(outputs),
                 voters=len(voters), rejected=dict(rejected))


def load_sources(solutions: pl.DataFrame, problem_ids: list[str]) -> dict[str, list[str]]:
    """Each problem's first voter_attempts C solutions, in solution_id order so a rerun picks the same voters."""
    rows = (solutions.filter(pl.col("problem_id").is_in(problem_ids))
            .sort("problem_id", "solution_id")
            .group_by("problem_id", maintain_order=True)
            .agg(pl.col("c").head(settings.differential.voter_attempts)))
    return dict(zip(rows["problem_id"].to_list(), rows["c"].to_list()))


def build_from_payload(payload: tuple[str, str, list[dict[str, str]], list[str]]) -> Suite:
    """One pool task: build_suite over a picklable payload."""
    return build_suite(*payload)


def main() -> None:
    options = CliApp.run(Settings).differential
    tasks = [json.loads(line) for line in options.tasks.read_text().splitlines()]
    solutions = pl.read_parquet(options.solutions)
    sources = load_sources(solutions, [task["task_id"].split("/")[1] for task in tasks])
    payloads = [(task["task_id"], task["arm_ref"], task["tests"], sources[task["task_id"].split("/")[1]])
                for task in tasks]
    options.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=options.workers) as pool, options.out.open("w") as sink:
        for position, suite in enumerate(pool.map(build_from_payload, payloads, chunksize=1)):
            sink.write(suite.model_dump_json() + "\n")
            sink.flush()
            print(position + 1, len(tasks), suite.task_id, "kept", suite.kept, "of", suite.candidates,
                  "voters", suite.voters, "outputs", suite.distinct_outputs, "usable", suite.usable, flush=True)


if __name__ == "__main__":
    main()
