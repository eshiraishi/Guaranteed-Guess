"""Runs inside the gg-ventura VM (/usr/bin/python3), reading one JSON problem per line on stdin and writing one JSON
result per line: the x86 source assembled for x86_64 and run under Rosetta 2 on each input, twice, so a
nondeterministic program is reported as having no answer. The Mac side (ggrpo.x86) compares the answers.

The host Mac runs macOS 27, where Rosetta no longer translates macOS binaries, so this is the only place the x86
source of a task can run at all."""
import json
import os
import resource
import subprocess
import sys
import tempfile

CPU_SECONDS = 5
RUN_TIMEOUT = 10
MAX_OUTPUT = 65536


def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_once(work, stdin):
    """The program's stdout, or None when it timed out, hit a signal or printed too much."""
    try:
        done = subprocess.run(["./a.out"], cwd=work, input=stdin + "\n", capture_output=True, text=True,
                              errors="replace", timeout=RUN_TIMEOUT, preexec_fn=set_limits)
    except subprocess.TimeoutExpired:
        return None
    if done.returncode < 0 or len(done.stdout) > MAX_OUTPUT:
        return None
    return "\n".join(line.rstrip() for line in done.stdout.rstrip().splitlines())


def answer(work, stdin):
    """The answer only when two runs agree, so undefined behaviour does not become an expected output."""
    first = run_once(work, stdin)
    if first is None:
        return None
    return first if run_once(work, stdin) == first else None


def handle(problem):
    with tempfile.TemporaryDirectory(prefix="gg-x86-") as tmp:
        with open(os.path.join(tmp, "src.s"), "w") as sink:
            sink.write(problem["x86"] + "\n")
        for stage, command in (("assemble", ["clang", "-arch", "x86_64", "-w", "-c", "-x", "assembler", "src.s",
                                             "-o", "src.o"]),
                               ("link", ["clang", "-arch", "x86_64", "-w", "src.o", "-o", "a.out", "-lm"])):
            built = subprocess.run(command, cwd=tmp, capture_output=True, text=True, errors="replace", timeout=60)
            if built.returncode:
                return {"task_id": problem["task_id"], "built": False, "stage": stage,
                        "error": built.stderr.strip()[-300:], "outputs": []}
        return {"task_id": problem["task_id"], "built": True,
                "outputs": [answer(tmp, stdin) for stdin in problem["inputs"]]}


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        result = handle(json.loads(line))
    except Exception as error:                      # one bad problem must not end the stream
        result = {"task_id": json.loads(line).get("task_id"), "built": False, "stage": "worker",
                  "error": repr(error)[:300], "outputs": []}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
