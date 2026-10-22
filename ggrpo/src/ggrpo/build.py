"""Build the MBPP-C O2 task pool in the macOS 13 VM. Each C function is compiled to x86 and arm with GG's commands
and gets a test driver. The reference arm must pass the driver, and a .s without the function must not link.
Writes build.out, which ggrpo.data reads."""
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import structlog
from pydantic import BaseModel

from ggrpo.harness import Verdict, check, run
from ggrpo.settings import Settings

settings = Settings()
log = structlog.get_logger("ggrpo")
# a C function definition; group 1 is its name (rl/data/build_humaneval.py DEF_RE)
definition = r"(?m)^[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*\([^;{)]*\)\s*\{"


class Task(BaseModel):
    """One MBPP-C O2 task: a row of build.out."""
    task_id: str
    driver: str
    x86: str | None
    arm_ref: str | None
    ref_verdict: Verdict
    ref_stderr: str
    neg_verdict: Verdict

    @staticmethod
    def compile(code: str, cmd: list[str]) -> str | None:
        """GG's compile: code.c -> code.s with cmd at build.opt; None if clang fails."""
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp, "code.c")
            src.write_text(code)
            p = run([*cmd, src, "-o", src.with_suffix(".s"), settings.build.opt], timeout=settings.harness.build_timeout)
            return src.with_suffix(".s").read_text() if p.returncode == 0 else None

    @classmethod
    def build(cls, task_id: str, code: str, driver: str) -> "Task":
        x86, arm = [cls.compile(code, cmd) for cmd in (settings.build.x86, settings.build.arm)]
        ref = check(arm or "", driver)
        return cls(task_id=task_id, driver=driver, x86=x86, arm_ref=arm, ref_verdict=ref.verdict,
                   ref_stderr=ref.stderr, neg_verdict=check("\t.text\n", driver).verdict)


def find_definitions(text: pl.Expr) -> pl.Expr:
    """Names of the C functions defined in text (used twice: translated unit and driver)."""
    return text.str.extract_all(definition).list.eval(pl.element().str.extract(definition, 1))


if __name__ == "__main__":   # pool workers import this module
    # Helpers that the driver (func_dep + test) shares with the translated unit (func_dep + func) are made weak, so the
    # predicted assembly's definitions win at link time.
    shared = (
        find_definitions(pl.concat_str("func_dep", "func"))
        .list.set_intersection(find_definitions(pl.concat_str("func_dep", "test")))
        .list.set_difference(pl.lit(["main", "func0"]))
        .list.sort()
    )
    prototype = pl.col("func").str.extract(r"(?:^|\n)([^\n;{}#]*\bfunc0\s*\([^{]*)\{", 1).str.strip_chars() + ";"
    source = pl.read_parquet(settings.build.source)
    tasks = (
        source.sort("index")
        .with_row_index("rank")
        .select(
            task_id=pl.format("mbpp/{}", "rank"),
            code=pl.concat_str("func_dep", "func", separator="\n"),
            driver=pl.concat_str(
                shared.list.eval(pl.format("#pragma weak {}\n", pl.element())).list.join(""),
                "func_dep", pl.lit("\n"), prototype, pl.lit("\n"), "test",
            ),
        )
        .drop_nulls("driver")   # no func0 definition to declare
    )
    with ProcessPoolExecutor(settings.build.workers) as pool:
        mbpp = pl.DataFrame(list(pool.map(Task.build, tasks["task_id"], tasks["code"], tasks["driver"], chunksize=4)))
    mbpp = mbpp.drop_nulls(["x86", "arm_ref"])   # clang failed on the C itself
    settings.build.out.parent.mkdir(parents=True, exist_ok=True)
    mbpp.write_parquet(settings.build.out)
    log.info("built", source=source.height, rows=mbpp.height,
             ref=Counter(mbpp["ref_verdict"].to_list()), neg=Counter(mbpp["neg_verdict"].to_list()))
