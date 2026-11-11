"""Harness verdicts on known inputs. Authoritative only inside the macOS 13 VM (the GG toolchain)."""
import glob
import platform

import polars as pl
import pytest

from ggrpo.build import Task
from ggrpo.harness import check, check_stdio, check_task
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


@pytest.mark.parametrize(("c", "outcomes"), [(add, "PP"), (constant, "PF"), (aborts, "FF")])
def test_check_stdio_runs_every_test(c, outcomes):
    assert check_stdio(Task.compile(c, settings.build.arm), tests).tests == outcomes


@pytest.mark.parametrize(("asm", "verdict"), [("", "empty"), ("\t.text\n", "link_err")])   # no main
def test_check_stdio_failures(asm, verdict):
    assert check_stdio(asm, tests).verdict == verdict


def test_check_task_dispatches_on_the_grader():
    assert check_task(task["arm_ref"], task["driver"], None).passed
    assert check_task(Task.compile(add, settings.build.arm), None, tests).passed


# Code after a literal section, shrunk from real references: ld asserts and writes a snapshot of about 116 MB
LD_ASSERTION = """\
	.section	__TEXT,__literal8,8byte_literals
	.long	7
_gcd:                                   ; @gcd
	.cfi_startproc
	mov	x8, x1
	sdiv	w9, w0, w1
	ret
	.cfi_endproc
	.cfi_startproc
	add	x29, sp, #48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	sub	x8, x29, #12
	str	x8, [sp, #16]
	adrp	x0, l_.str@PAGE
	str	x8, [sp]
Lloh2:
	adrp	x0, l_.str.1@PAGE
Lloh3:
	add	x0, x0, l_.str.1@PAGEOFF
	bl	_printf
	.cfi_endproc
"""


@pytest.mark.skipif(not platform.mac_ver()[0].startswith("13."), reason="only the VM's ld64 asserts on this")
def test_ld_assertion_leaves_no_snapshot():
    before = set(glob.glob("/private/tmp/a.out-*.ld-snapshot*"))
    result = check_stdio(LD_ASSERTION, [{"input": "", "output": ""}])
    assert result.verdict == "link_err"
    assert "Assertion failed" in result.stderr
    assert set(glob.glob("/private/tmp/a.out-*.ld-snapshot*")) == before
