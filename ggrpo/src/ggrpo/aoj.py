"""AOJ pool: CodeNet's accepted C solutions for Aizu problems x AOJ's official test cases, one row per
(problem, solution). Uses only AOJ's documented APIs (developers.u-aizu.ac.jp): the Main API problem list and the
Data API test cases. Every response is saved under aoj.cache and never requested again, so a stopped run resumes.
Requests go one at a time, aoj.delay seconds apart, and back off on 429 and 5xx.

CodeNet renumbers problems (p00481 is AOJ 0558), so problems are matched by title and the match is kept only when
CodeNet's sample input is one of AOJ's sample inputs (CodeNet's metadata repeats the input as the output, Nov 10)."""
import json
import time

import httpx
import polars as pl
import structlog
from pydantic_settings import CliApp
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
client = httpx.Client(headers={"User-Agent": "ggrpo-research"}, timeout=120, follow_redirects=True)


def is_transient(error: BaseException) -> bool:
    return isinstance(error, httpx.TransportError) or (
        isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (429, 500, 502, 503, 504)
    )


@retry(retry=retry_if_exception(is_transient), wait=wait_exponential(min=10, max=600), stop=stop_after_attempt(8),
       reraise=True)
def request(url: str) -> bytes:
    time.sleep(settings.aoj.delay)
    response = client.get(url)
    response.raise_for_status()
    return response.content


def fetch(url: str, name: str) -> str | None:
    """Body of url, or None when AOJ answers 404 (no such data). Both outcomes are cached under aoj.cache/name."""
    path, missing = settings.aoj.cache / name, settings.aoj.cache / f"{name}.404"
    if missing.exists():
        return None
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            body = request(url)
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                raise
            missing.touch()
            return None
        part = path.with_name(f"{path.name}.part")
        part.write_bytes(body)
        part.replace(path)
    return path.read_bytes().decode()   # strict UTF-8: a test that is not text stops the run


def fetch_tests(problem_id: str) -> list[dict[str, str]] | None:
    """All official test cases of an AOJ problem, as {input, output}; None when AOJ has no judge data for it."""
    header = fetch(f"{settings.aoj.testcases}/{problem_id}/header", f"{problem_id}/header.json")
    if header is None:
        return None
    return [
        {kind: fetch(f"{settings.aoj.testcases}/{problem_id}/{case['serial']}/{part}",
                     f"{problem_id}/{case['serial']}.{part}")
         for kind, part in [("input", "in"), ("output", "out")]}
        for case in json.loads(header)["headers"]
    ]


def fetch_samples(problem_id: str) -> list[tuple[str, str]]:
    samples = fetch(f"{settings.aoj.testcases}/samples/{problem_id}", f"{problem_id}/samples.json")
    return [(case["in"].strip(), case["out"].strip()) for case in json.loads(samples or "[]")]


size = 100_000
aoj = pl.DataFrame(json.loads(fetch(f"{settings.aoj.problems}?page=0&size={size}", "problems.json"))).select(
    aoj_id="id", title=pl.col("name").str.strip_chars()
)
if aoj.height == size:
    raise ValueError("the AOJ problem list filled one page: raise size")

solutions = (
    pl.scan_parquet(settings.aoj.codenet.download() / "C")
    .filter(original_language=settings.aoj.language, status="Accepted")   # Aizu's label; AtCoder's is "C (GCC ...)"
    .select(codenet_id="p_id", solution_id="s_id", c="code")
)
codenet = (
    pl.scan_parquet(settings.aoj.metadata.download() / "data")
    .join(solutions.select("codenet_id").unique(), left_on="id", right_on="codenet_id", how="semi")
    .select(
        codenet_id="id",
        title=pl.col("slug_name").str.strip_chars(),
        sample_input=pl.col("test_cases").list.first().list.first().str.strip_chars(),   # [input, input]: see the docstring
    )
    .collect()
)
if settings.aoj.only:
    only = pl.concat([pl.read_ndjson(path).select(codenet_id=pl.col("task_id").str.extract(r"^codenet/([^/]+)/", 1))
                      for path in settings.aoj.only])["codenet_id"].unique()
    codenet = codenet.filter(pl.col("codenet_id").is_in(only.implode()))
    log.info("only", files=[str(path) for path in settings.aoj.only], problems=only.len(), with_metadata=codenet.height)
candidates = codenet.join(aoj, on="title", how="inner")
verified = [sample_input in [case_input for case_input, _ in fetch_samples(aoj_id)]
            for aoj_id, sample_input in candidates[["aoj_id", "sample_input"]].iter_rows()]
mapping = candidates.filter(pl.Series(verified))
if mapping.height == 0:
    raise ValueError(f"no CodeNet problem matched AOJ by title and sample input ({candidates.height} title matches)")
if mapping["codenet_id"].is_duplicated().any() or mapping["aoj_id"].is_duplicated().any():
    raise ValueError(f"ambiguous mapping: {mapping.filter(pl.col('codenet_id').is_duplicated())}")
log.info("mapped", codenet=codenet.height, title_matches=candidates["codenet_id"].n_unique(), verified=mapping.height)

tests = pl.DataFrame(
    [{"aoj_id": problem_id, "tests": fetch_tests(problem_id)} for problem_id in mapping["aoj_id"]],
    schema={"aoj_id": pl.String, "tests": pl.List(pl.Struct({"input": pl.String, "output": pl.String}))},
)
problems = mapping.join(tests, on="aoj_id").filter(
    pl.col("tests").is_not_null(),
    # a program that prints a constant must not pass (the same rule as Codeforces)
    pl.col("tests").list.eval(pl.element().struct.field("output")).list.n_unique() >= 2,
)
log.info("tests", problems=mapping.height, no_judge_data=tests["tests"].is_null().sum(), kept=problems.height)

pool = (   # one row per problem: a row per solution stalls the Mac (39 GB in swap, Nov 11) on the GBs of tests
    problems.lazy()
    .join(solutions.group_by("codenet_id").agg(solutions=pl.struct("solution_id", "c").sort_by("solution_id")),
          on="codenet_id", how="inner")
    .select(pl.col("aoj_id").alias("problem_id"), "codenet_id", "tests", "solutions")
    .sort("problem_id")
    .collect()
)
settings.aoj.out.parent.mkdir(parents=True, exist_ok=True)
pool.write_parquet(settings.aoj.out)
log.info("wrote", path=str(settings.aoj.out), problems=pool.height, solutions=pool["solutions"].list.len().sum())
