"""Assemble a predicted .s, link it with the task's C test driver, run it sandboxed, return a verdict.
The prediction is the only definition of the function, so an empty or truncated one cannot pass.

Verdicts: pass | empty (nothing extracted) | assemble_err | link_err | crash (killed by a signal, e.g. SIGABRT
from a failed assert) | timeout | wrong_output (non-zero exit) | infra_err (the test driver itself did not build)."""
import hashlib
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field

from ggrpo.settings import Settings

settings = Settings()
run = partial(subprocess.run, capture_output=True, text=True, errors="replace")
# ld writes one of these, about 116 MB, on every assertion; malformed model output asserts often
linker_snapshot = re.compile(r"^A linker snapshot was created at:\n\t(/(?:private/)?tmp/a\.out-[\d-]+\.ld-snapshot(?:-\d+)?)$",
                             re.MULTILINE)

Verdict = Literal["pass", "empty", "assemble_err", "link_err", "crash", "timeout", "wrong_output", "infra_err"]


class Result(BaseModel):
    verdict: Verdict
    returncode: int | None = None
    stderr: str = ""
    seconds: float = 0.0
    tests: str = ""            # stdio mode: P or F per test run, in order (the run ends at a timeout); else empty

    @computed_field
    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def set_limits() -> None:
    """preexec_fn of the test binary: CPU time, file size, no core dumps."""
    resource.setrlimit(resource.RLIMIT_CPU, (settings.harness.cpu_seconds, settings.harness.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (settings.harness.max_file_bytes, settings.harness.max_file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def link(work: Path, asm: str, objects: list[Path], start: float) -> Result | None:
    """Assemble work/pred.s and link it with objects into work/a.out: the failure, or None when a.out was built."""
    (work / "pred.s").write_text(asm + "\n")
    for failure, cmd in [
        ("assemble_err", [*settings.harness.asm, "pred.s", "-o", "pred.o"]),
        ("link_err", [*settings.harness.cc, "pred.o", *objects, "-o", "a.out", *settings.harness.ldflags]),
    ]:
        try:
            p = run(cmd, cwd=work, timeout=settings.harness.build_timeout)
        except subprocess.TimeoutExpired:
            return Result(verdict=failure, stderr="timeout", seconds=time.monotonic() - start)
        if p.returncode:
            if snapshot := linker_snapshot.search(p.stderr):
                shutil.rmtree(snapshot.group(1))
            return Result(verdict=failure, returncode=p.returncode, stderr=p.stderr[-settings.harness.stderr_tail:],
                          seconds=time.monotonic() - start)
    return None


def execute(work: Path, stdin: str | None = None, program: str = "./a.out") -> subprocess.CompletedProcess | None:
    """Run a program in work sandboxed, with the resource limits; None when it hits the wall-clock timeout."""
    try:
        return run(["sandbox-exec", "-f", settings.harness.sandbox.resolve(), "-D", f"WORKDIR={work}", program],
                   cwd=work, timeout=settings.harness.run_timeout, preexec_fn=set_limits, input=stdin)
    except subprocess.TimeoutExpired:
        return None


def normalize(text: str) -> str:
    """Output as judges compare it: trailing whitespace of every line and of the whole text is ignored."""
    return "\n".join(line.rstrip() for line in text.rstrip().splitlines())


def check_stdio(asm: str, tests: list[dict[str, str]]) -> Result:
    """stdio mode: the prediction is a whole program (it defines main) and is linked alone. It passes when its stdout
    equals each test's output up to trailing whitespace. The exit status is ignored, as for a main without a return;
    a signal is a crash. Test inputs are stored stripped, so each gets its final newline back. Every test runs, so
    Result.tests records each outcome (prereg.md §2, Recording), except that a timeout ends the run; the verdict is the
    first failure's."""
    start = time.monotonic()
    if not asm.strip():
        return Result(verdict="empty")
    outcomes, failure = "", None
    with tempfile.TemporaryDirectory(prefix="gg-run-") as tmp:
        work = Path(tmp).resolve()
        if failure := link(work, asm, [], start):
            return failure
        for test in tests:
            p = execute(work, test["input"] + "\n")
            timed_out = p is None or -p.returncode == signal.SIGXCPU   # wall clock, or the CPU limit's signal
            passed = not timed_out and p.returncode >= 0 and normalize(p.stdout) == normalize(test["output"])
            outcomes += "P" if passed else "F"
            if not passed and failure is None:
                if p is None:
                    failure = Result(verdict="timeout")
                else:
                    verdict = "wrong_output" if p.returncode >= 0 else "timeout" if timed_out else "crash"
                    failure = Result(verdict=verdict, returncode=p.returncode,
                                     stderr=p.stderr[-settings.harness.stderr_tail:])
            if timed_out:
                break
    result = failure or Result(verdict="pass")
    return result.model_copy(update={"tests": outcomes, "seconds": time.monotonic() - start})


def check_task(asm: str, driver: str | None, tests: list[dict[str, str]] | None) -> Result:
    """A task has either a C test driver (MBPP-C, HumanEval-C) or stdio tests (CodeNet)."""
    return check(asm, driver) if driver is not None else check_stdio(asm, tests)


def check(asm: str, driver: str) -> Result:
    start = time.monotonic()
    if not asm.strip():
        return Result(verdict="empty")

    # Compile each test driver once, cached by the hash of compiler + source.
    driver_object = (
        settings.harness.cache / f"{hashlib.sha256(' '.join([*settings.harness.cc, driver]).encode()).hexdigest()[:24]}.o"
    ).resolve()
    if not driver_object.exists():
        settings.harness.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=settings.harness.cache) as tmp:   # same filesystem -> atomic replace
            Path(tmp, "src.c").write_text(driver)
            try:
                run([*settings.harness.cc, "-c", "src.c", "-o", "src.o"], cwd=tmp, check=True,
                    timeout=settings.harness.build_timeout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return Result(verdict="infra_err", stderr=f"{e}\n{e.stderr}"[-settings.harness.stderr_tail:])
            Path(tmp, "src.o").replace(driver_object)

    with tempfile.TemporaryDirectory(prefix="gg-run-") as tmp:
        work = Path(tmp).resolve()
        if failure := link(work, asm, [driver_object], start):
            return failure
        p = execute(work)
    if p is None:
        return Result(verdict="timeout", seconds=time.monotonic() - start)

    match p.returncode:
        case 0:
            verdict = "pass"
        case returncode if returncode > 0:
            verdict = "wrong_output"
        case returncode if -returncode == signal.SIGXCPU:
            verdict = "timeout"
        case _:
            verdict = "crash"
    return Result(verdict=verdict, returncode=p.returncode, stderr=p.stderr[-settings.harness.stderr_tail:],
                  seconds=time.monotonic() - start)
