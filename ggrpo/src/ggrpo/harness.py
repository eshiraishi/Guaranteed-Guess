"""Assemble a predicted .s, link it with the task's C test driver, run it sandboxed, return a verdict.
The prediction is the only definition of the function, so an empty or truncated one cannot pass.

Verdicts: pass | empty (nothing extracted) | assemble_err | link_err | crash (killed by a signal, e.g. SIGABRT
from a failed assert) | timeout | wrong_output (non-zero exit) | infra_err (the test driver itself did not build)."""
import hashlib
import resource
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

Verdict = Literal["pass", "empty", "assemble_err", "link_err", "crash", "timeout", "wrong_output", "infra_err"]


class Result(BaseModel):
    verdict: Verdict
    returncode: int | None = None
    stderr: str = ""
    seconds: float = 0.0

    @computed_field
    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def set_limits() -> None:
    """preexec_fn of the test binary: CPU time, file size, no core dumps."""
    resource.setrlimit(resource.RLIMIT_CPU, (settings.harness.cpu_seconds, settings.harness.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (settings.harness.max_file_bytes, settings.harness.max_file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


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
        (work / "pred.s").write_text(asm + "\n")
        for failure, cmd in [
            ("assemble_err", [*settings.harness.asm, "pred.s", "-o", "pred.o"]),
            ("link_err", [*settings.harness.cc, "pred.o", driver_object, "-o", "a.out", *settings.harness.ldflags]),
        ]:
            try:
                p = run(cmd, cwd=work, timeout=settings.harness.build_timeout)
            except subprocess.TimeoutExpired:
                return Result(verdict=failure, stderr="timeout", seconds=time.monotonic() - start)
            if p.returncode:
                return Result(verdict=failure, returncode=p.returncode, stderr=p.stderr[-settings.harness.stderr_tail:],
                              seconds=time.monotonic() - start)
        try:
            p = run(["sandbox-exec", "-f", settings.harness.sandbox.resolve(), "-D", f"WORKDIR={work}", "./a.out"],
                    cwd=work, timeout=settings.harness.run_timeout, preexec_fn=set_limits)
        except subprocess.TimeoutExpired:
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
