"""AnghaBench's C files as parquet (Mac, --extra data), for ggrpo.decontam: GG's AnghaBench documents have no C in the
Hugging Face cache. Reads the GitHub tarball at the commit in decontam.angha_url; nothing is extracted to disk."""
import tarfile
import urllib.request

import polars as pl
import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")

with urllib.request.urlopen(settings.decontam.angha_url) as response, tarfile.open(fileobj=response,
                                                                                   mode="r|gz") as archive:
    files = [(f"AnghaBench/{member.name.split('/', 1)[1]}", archive.extractfile(member).read().decode(errors="replace"))
             for member in archive if member.isfile() and member.name.endswith(".c")]
settings.decontam.angha_c.parent.mkdir(parents=True, exist_ok=True)
pl.DataFrame(files, schema={"file": pl.String, "c": pl.String}, orient="row").write_parquet(settings.decontam.angha_c)
log.info("wrote", path=str(settings.decontam.angha_c), files=len(files))
