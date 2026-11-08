"""Compiles the planted copies of ggrpo.canary to x86 in the macOS 13 VM (the GG toolchain), each at its own -O level.
Writes decontam.canary_x86; a copy that clang rejects has no x86."""
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")


def compile_copy(row: tuple[str, str, str, str]) -> tuple[str, str, str | None]:
    base, variant, flag, code = row
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp, "code.c")
        source.write_text(code)
        try:
            result = subprocess.run([*settings.build.x86, str(source), "-o", str(source.with_suffix(".s")), flag],
                                    capture_output=True, check=False, timeout=settings.harness.build_timeout)
        except subprocess.TimeoutExpired:
            return base, variant, None
        return base, variant, source.with_suffix(".s").read_text() if result.returncode == 0 else None


if __name__ == "__main__":   # pool workers import this module
    planted = pl.read_parquet(settings.decontam.canary_c).select("base", "variant", "flag", "c")
    with ProcessPoolExecutor(settings.build.workers) as pool:
        rows = list(pool.map(compile_copy, planted.iter_rows(), chunksize=32))
    built = pl.DataFrame(rows, schema={"base": pl.String, "variant": pl.String, "x86": pl.String}, orient="row")
    built.write_parquet(settings.decontam.canary_x86)
    log.info("wrote", path=str(settings.decontam.canary_x86), copies=built.height,
             built=built["x86"].is_not_null().sum())
