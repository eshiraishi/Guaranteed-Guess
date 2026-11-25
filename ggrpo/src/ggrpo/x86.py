"""Add the x86 source as the third voter on every kept differential input (prereg.md §4). The host Mac runs macOS 27,
where Rosetta 2 no longer translates macOS binaries, so the source runs in the gg-ventura VM through one ssh pipe to
ops/vm/x86_worker.py: one problem per line, one result per line.

This is a pass over an existing suite file rather than part of building it. Every voter has to agree on a kept input,
so filtering afterwards gives the same set as voting during the build. A problem whose source fails its own sample
tests keeps the reference-and-solutions verdict and is counted, as the prereg says."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import structlog
from pydantic import BaseModel
from pydantic_settings import CliApp

from ggrpo.harness import normalize
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")
x86_block = re.compile(r"(?s)```x86asm\n(.*)\n```")


class Confirmed(BaseModel):
    """One problem's suite after the x86 vote."""
    task_id: str
    built: bool
    source_ok: bool            # the x86 source answers every sample test correctly
    kept_before: int
    kept: int
    dropped: int
    tests: list[dict[str, str]]
    samples: int
    distinct_outputs: int
    voters: int


def read_x86(task: dict) -> str:
    """The x86 assembly the task asks the model to translate."""
    found = x86_block.search(task["prompt"][0]["content"])
    if found is None:
        raise ValueError(f"{task['task_id']}: no x86asm block in the prompt")
    return found.group(1)


def start_worker(host: str, worker: Path) -> subprocess.Popen:
    """Copy the worker into the VM and open one long-lived pipe to it."""
    options = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    subprocess.run(["scp", *options, str(worker), f"{host}:/tmp/x86_worker.py"], check=True, capture_output=True)
    return subprocess.Popen(["ssh", *options, host, "/usr/bin/python3", "-u", "/tmp/x86_worker.py"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def ask(worker: subprocess.Popen, task_id: str, source: str, inputs: list[str]) -> dict:
    """One round trip: the source's answer on each input, or None where it has none."""
    worker.stdin.write(json.dumps({"task_id": task_id, "x86": source, "inputs": inputs}) + "\n")
    worker.stdin.flush()
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f"{task_id}: the x86 worker closed the pipe")
    return json.loads(line)


def confirm(suite: dict, source: str, answers: dict) -> Confirmed:
    """Drop every kept input the x86 source does not answer exactly as the suite records."""
    samples = suite["tests"][:suite["samples"]]
    kept = suite["tests"][suite["samples"]:]
    outputs = answers["outputs"]
    source_ok = answers["built"] and all(
        given == normalize(sample["output"]) for given, sample in zip(outputs[:len(samples)], samples))
    if not source_ok:
        return Confirmed(task_id=suite["task_id"], built=answers["built"], source_ok=False,
                         kept_before=len(kept), kept=len(kept), dropped=0, tests=suite["tests"],
                         samples=suite["samples"], distinct_outputs=suite["distinct_outputs"],
                         voters=suite["voters"])
    agreed = [test for test, given in zip(kept, outputs[len(samples):]) if given == test["output"]]
    distinct = {test["output"] for test in agreed} | {normalize(sample["output"]) for sample in samples}
    return Confirmed(task_id=suite["task_id"], built=True, source_ok=True, kept_before=len(kept), kept=len(agreed),
                     dropped=len(kept) - len(agreed), tests=samples + agreed, samples=suite["samples"],
                     distinct_outputs=len(distinct), voters=suite["voters"] + 1)


def main() -> None:
    options = CliApp.run(Settings).x86
    suites = [json.loads(line) for line in options.suites.read_text().splitlines()]
    sources = {task["task_id"]: read_x86(task)
               for task in (json.loads(line) for line in options.tasks.read_text().splitlines())}
    worker = start_worker(options.host, options.worker)
    counts: Counter = Counter()
    options.out.parent.mkdir(parents=True, exist_ok=True)
    with options.out.open("w") as sink:
        for position, suite in enumerate(suites):
            inputs = [test["input"] for test in suite["tests"]]
            answers = ask(worker, suite["task_id"], sources[suite["task_id"]], inputs)
            result = confirm(suite, sources[suite["task_id"]], answers)
            sink.write(result.model_dump_json() + "\n")
            sink.flush()
            counts["problems"] += 1
            counts["source_failed"] += int(not result.source_ok)
            counts["dropped_inputs"] += result.dropped
            counts["usable"] += int(result.kept >= settings.differential.min_kept and result.distinct_outputs >= 2)
            if (position + 1) % 25 == 0:
                print(position + 1, "of", len(suites), dict(counts), flush=True)
    worker.stdin.close()
    worker.wait(timeout=60)
    log.info("x86", suites=str(options.suites), out=str(options.out), **dict(counts))


if __name__ == "__main__":
    main()
