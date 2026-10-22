"""Functional-correctness harness: assemble a predicted .s, link it against a task's
test driver, run it, and return a verdict.

Unlike eval/bringup-bench/Makefile, the predicted assembly is the ONLY definition of
the translated function: the C source of the function is never compiled into the
test binary, so a missing/empty prediction cannot silently pass.
"""
from __future__ import annotations

import enum
import hashlib
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


class Verdict(str, enum.Enum):
    PASS = "pass"
    EMPTY = "empty"                # nothing extracted / truncated generation
    ASSEMBLE_ERR = "assemble_err"
    LINK_ERR = "link_err"          # undefined / duplicate symbols etc.
    CRASH = "crash"                # killed by a signal (SIGSEGV, SIGBUS, SIGABRT from assert...)
    TIMEOUT = "timeout"
    WRONG_OUTPUT = "wrong_output"  # non-zero exit or stdout mismatch
    INFRA_ERR = "infra_err"        # harness-side failure (test driver did not build)


@dataclass
class Toolchain:
    name: str
    cc: list                       # compiler driver for support sources + linking
    asm: list                      # assembler invocation for the predicted .s
    run_prefix: list = field(default_factory=list)  # e.g. qemu-arm -L sysroot
    ldflags: list = field(default_factory=lambda: ["-lm"])
    sandbox: bool = False          # macOS sandbox-exec around the test binary


TOOLCHAINS = {
    # Apple arm64 (GG ARMv8 data is Mach-O, produced by `clang -arch arm64`).
    "macos-arm64": Toolchain(
        name="macos-arm64",
        cc=["clang", "-arch", "arm64", "-w"],
        asm=["clang", "-arch", "arm64", "-c", "-x", "assembler"],
        sandbox=True,
    ),
    # ARMv5 (GG ARMv5 data is arm-linux-gnueabi-gcc output), run under QEMU user mode.
    "linux-armv5": Toolchain(
        name="linux-armv5",
        cc=["arm-linux-gnueabi-gcc", "-w"],
        asm=["arm-linux-gnueabi-gcc", "-c", "-x", "assembler"],
        run_prefix=["qemu-arm", "-L", "/usr/arm-linux-gnueabi"],
    ),
}


def default_toolchain():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return TOOLCHAINS["macos-arm64"]
    return TOOLCHAINS["linux-armv5"]


@dataclass
class Result:
    verdict: Verdict
    returncode: int | None = None
    stderr_tail: str = ""
    seconds: float = 0.0
    frac_passed: float | None = None   # drivers that print "GG_PASSED k/n" (ExeBench)

    @property
    def passed(self):
        return self.verdict == Verdict.PASS

    def to_dict(self):
        return {"verdict": self.verdict.value, "passed": self.passed, "returncode": self.returncode,
                "stderr_tail": self.stderr_tail, "seconds": round(self.seconds, 4),
                "frac_passed": self.passed if self.frac_passed is None else self.frac_passed}


def _tail(b, n=600):
    s = b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "")
    return s[-n:]


def _limits(cpu_seconds):
    def fn():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return fn


def _run(cmd, cwd, timeout, cpu_seconds=None, stdin=None):
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL if stdin is None else subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=_limits(cpu_seconds or timeout))
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout)
        return proc.returncode, out, err, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return proc.returncode, out, err, True


class Harness:
    """Stateless apart from a content-addressed cache of compiled test drivers."""

    def __init__(self, toolchain=None, cache_dir=None, build_timeout=30, run_timeout=10, cpu_seconds=5):
        self.tc = toolchain or default_toolchain()
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), f"gg-harness-cache-{self.tc.name}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.build_timeout = build_timeout
        self.run_timeout = run_timeout
        self.cpu_seconds = cpu_seconds
        self.sandbox_profile = os.path.join(HERE, "sandbox.sb")

    def _driver_objects(self, task):
        """Compile the task's support sources (test main, wrappers, helper libs) once and cache."""
        objs = []
        for fname, src in sorted(task["support_src"].items()):
            key = hashlib.sha256((self.tc.name + fname + src + " ".join(task.get("cflags", []))).encode()).hexdigest()[:24]
            obj = os.path.join(self.cache_dir, f"{key}.o")
            if not os.path.exists(obj):
                ext = os.path.splitext(fname)[1] or ".c"
                with tempfile.TemporaryDirectory() as d:
                    path = os.path.join(d, "src" + ext)
                    with open(path, "w") as f:
                        f.write(src)
                    cc = self.tc.cc + ["-I", os.path.join(HERE, "include")]
                    if ext in (".cpp", ".cc"):
                        cc = cc + ["-std=c++17"]
                    rc, _, err, to = _run(cc + task.get("cflags", []) + ["-c", path, "-o", obj + ".tmp"],
                                          d, self.build_timeout)
                    if rc != 0 or to:
                        raise RuntimeError(f"driver {fname} failed to build: {_tail(err)}")
                    os.replace(obj + ".tmp", obj)
            objs.append(obj)
        return objs

    def check(self, asm, task):
        """task: {"support_src": {name: source}, "cflags": [...], "ldflags": [...],
                  "expected_stdout": str | None, "stdin": str | None, "cxx": bool}"""
        t0 = time.time()
        if not asm or not asm.strip():
            return Result(Verdict.EMPTY, seconds=time.time() - t0)
        try:
            objs = self._driver_objects(task)
        except Exception as e:
            return Result(Verdict.INFRA_ERR, stderr_tail=str(e)[-600:], seconds=time.time() - t0)

        work = tempfile.mkdtemp(prefix="gg-run-")
        try:
            with open(os.path.join(work, "pred.s"), "w") as f:
                f.write(asm if asm.endswith("\n") else asm + "\n")
            for name, content in task.get("files", {}).items():  # runtime inputs (BringUp: words, *.cnf, ...)
                with open(os.path.join(work, os.path.basename(name)), "w") as f:
                    f.write(content)
            rc, _, err, to = _run(self.tc.asm + ["pred.s", "-o", "pred.o"], work, self.build_timeout)
            if rc != 0 or to:
                return Result(Verdict.ASSEMBLE_ERR, rc, _tail(err), time.time() - t0)

            linker = self.tc.cc
            if task.get("cxx"):
                linker = [c.replace("clang", "clang++").replace("gcc", "g++") if i == 0 else c
                          for i, c in enumerate(self.tc.cc)]
            rc, _, err, to = _run(linker + ["pred.o"] + objs + ["-o", "a.out"] + self.tc.ldflags + task.get("ldflags", []),
                                  work, self.build_timeout)
            if rc != 0 or to:
                return Result(Verdict.LINK_ERR, rc, _tail(err), time.time() - t0)

            cmd = self.tc.run_prefix + ["./a.out"] + task.get("args", [])
            if self.tc.sandbox and shutil.which("sandbox-exec"):
                cmd = ["sandbox-exec", "-f", self.sandbox_profile, "-D", f"WORKDIR={os.path.realpath(work)}"] + cmd
            stdin = task.get("stdin")
            rt = task.get("run_timeout", self.run_timeout)
            cpu = rt if "run_timeout" in task else self.cpu_seconds
            rc, out, err, to = _run(cmd, work, rt, cpu,
                                    stdin=stdin.encode() if stdin is not None else None)
            dt = time.time() - t0
            m = re.search(rb"GG_PASSED (\d+)/(\d+)", out or b"")
            frac = int(m.group(1)) / max(1, int(m.group(2))) if m else None
            if to:
                return Result(Verdict.TIMEOUT, rc, _tail(err), dt)
            if rc is not None and rc < 0:
                # SIGXCPU from RLIMIT_CPU is a timeout, other signals are crashes (assert -> SIGABRT).
                v = Verdict.TIMEOUT if -rc == signal.SIGXCPU else Verdict.CRASH
                return Result(v, rc, _tail(err), dt)
            if rc != 0 and task.get("check_rc", True):
                return Result(Verdict.WRONG_OUTPUT, rc, _tail(err), dt, frac)
            exp = task.get("expected_stdout")
            if exp is not None and out.decode("utf-8", "replace") != exp:
                return Result(Verdict.WRONG_OUTPUT, rc, "stdout mismatch", dt)
            return Result(Verdict.PASS, rc, "", dt)
        finally:
            shutil.rmtree(work, ignore_errors=True)
