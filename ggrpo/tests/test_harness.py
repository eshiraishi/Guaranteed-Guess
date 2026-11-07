"""Harness verdicts on known inputs. Authoritative only inside the macOS 13 VM (the GG toolchain)."""
import platform

import polars as pl
import pytest

from ggrpo.build import Task
from ggrpo.harness import check, check_stdio
from ggrpo.settings import Settings

settings = Settings()
pytestmark = pytest.mark.skipif(platform.machine() != "arm64" or platform.system() != "Darwin",
                                reason="needs the macOS arm64 toolchain")
task, other = pl.read_ndjson(settings.data.eval).filter(pl.col("task_id").is_in(["humaneval/3", "humaneval/4"])).sort("task_id").to_dicts()


@pytest.mark.parametrize(
    ("asm", "verdict"),
    [
        (task["arm_ref"], "pass"),
        ("", "empty"),
        ("not assembly", "assemble_err"),
        ("\t.text\n", "link_err"),          # the driver does not define the function itself
        (other["arm_ref"], "crash"),        # another problem's func0 links but fails the asserts
    ],
)
def test_check(asm, verdict):
    assert check(asm, task["driver"]).verdict == verdict


tests = [{"input": "1 2", "output": "3"}, {"input": "40 2", "output": "42"}]
add = "#include <stdio.h>\nint main(void) { int a, b; scanf(\"%d %d\", &a, &b); printf(\"%d  \\n\\n\", a + b); }\n"
constant = "#include <stdio.h>\nint main(void) { puts(\"3\"); }\n"
aborts = "#include <stdlib.h>\nint main(void) { abort(); }\n"


@pytest.mark.parametrize(
    ("c", "verdict"),
    [
        (add, "pass"),                      # trailing spaces and blank lines do not count
        (constant, "wrong_output"),         # right on the first test only
        (aborts, "crash"),
    ],
)
def test_check_stdio(c, verdict):
    assert check_stdio(Task.compile(c, settings.build.arm), tests).verdict == verdict


@pytest.mark.parametrize(("asm", "verdict"), [("", "empty"), ("\t.text\n", "link_err")])   # no main
def test_check_stdio_failures(asm, verdict):
    assert check_stdio(asm, tests).verdict == verdict
