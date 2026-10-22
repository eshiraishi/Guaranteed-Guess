"""Harness verdicts on known inputs. Authoritative only inside the macOS 13 VM (the GG toolchain)."""
import platform

import polars as pl
import pytest

from ggrpo.harness import check
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
