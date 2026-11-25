"""Codeforces task files (Mac, --extra data), after ggrpo.codeforces_build (VM) and the decontamination run of its
candidates (ggrpo.decontam with --decontam.c_items codeforces.candidates). Each problem gets the first reason that
applies, in this order, or "kept"; no reason reads a model's output:
- no_build: no solution's x86 and arm built;
- layout_only: none passed every example and official test in the VM, but one matched every test up to whitespace
  or letter case (Codeforces' checkers accept that; the scorers compare lines exactly, so no task is built);
- reference_fails: none matched every test even so;
- slow: every one that passed needs more than codeforces.max_cpu CPU seconds on some test;
- host_fails: none of the rest passes again on this host, where ggrpo.official scores, within the same margin. The
  first that does, in pick order, is the reference;
- multi_answer: among the built solutions that finished every test (no crash, timeout or link error) and use no
  MinGW-only idiom (they break off Windows), >= 2 and >= codeforces.dissent of them printed a wrong output on some
  test: several valid answers, or a program that depends on its platform. An output with the same tokens (S) or the
  same tokens up to letter case (K) agrees: Codeforces' checkers accept both, and a translation prints its reference's
  layout and case. With fewer than 3 such solutions agreement
  is not checked; the problem is kept and flagged;
- contaminated: the reference's task has a drop pair in codeforces.decontam (asm, c or exact rule);
- shared_x86: the reference's x86 prompt is also a CodeNet task's or another kept Codeforces task's.
Writes the task files per split (sample tests = the statement's examples), codeforces.eval (official tests of the kept
problems, read by ggrpo.official) and codeforces.drops (one row per problem). Tasks whose prompt and reference exceed
the 32,768-token context are kept and flagged: prereg §3 counts them as failures for every arm."""
import re
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import polars as pl
import structlog
from huggingface_hub import hf_hub_download
from pydantic_settings import CliApp
from tokenizers import Tokenizer

from ggrpo.codeforces_build import run_tests
from ggrpo.settings import Settings

log = structlog.get_logger("ggrpo")
mingw = re.compile(r"%I64|__int64|<windows\.h>|<conio\.h>")   # MinGW-only: wrong or no build off Windows
context = 32768   # the base model's positions (prereg §3's over-length rule)


def check_on_host(candidates: list[tuple[str, str]], examples: list[dict[str, str]], tests: list[dict[str, str]],
                  max_cpu: float) -> tuple[str | None, float | None]:
    """The first (solution id, arm) that passes every example and official test on this host, each test within
    max_cpu CPU seconds, and its slowest test's CPU seconds; (None, None) when none does."""
    for solution_id, arm in candidates:
        (on_examples, examples_cpu), (on_tests, tests_cpu) = run_tests(arm, examples), run_tests(arm, tests)
        cpu = max(examples_cpu, tests_cpu)
        if on_examples == "P" * len(examples) and on_tests == "P" * len(tests) and cpu <= max_cpu:
            return solution_id, cpu
    return None, None


def count_agreement(outcomes: list[str | None], sources: list[str], tests: int) -> tuple[int, int]:
    """(finishing, dissenting) among one problem's built solutions. outcomes holds each solution's results on the
    examples and the official tests together (None when its arm did not build). A solution finishes when it ran all
    `tests` without a crash, timeout or link error and its source has no MinGW-only idiom; it dissents when it printed
    a wrong output on some test (W; the same tokens, S, or the same tokens up to letter case, K, agree)."""
    finishing = [result for result, source in zip(outcomes, sources, strict=True)
                 if result is not None and len(result) == tests and set(result) <= {"P", "S", "K", "W"}
                 and not mingw.search(source)]
    return len(finishing), sum("W" in result for result in finishing)


def main() -> None:
    settings = CliApp.run(Settings)
    options = settings.codeforces
    problems = pl.read_parquet(options.out).with_columns(n_examples=pl.col("examples").list.len(),
                                                         n_tests=pl.col("tests").list.len())
    sources = problems.select("problem_id", "solutions").explode("solutions").select(
        "problem_id", solution_id=pl.col("solutions").struct.field("solution_id"), c=pl.col("solutions").struct.field("c")
    )
    built = (
        pl.read_parquet(options.built / "part-*.parquet")
        .join(sources, on=["problem_id", "solution_id"], validate="1:1")
        .join(problems.select("problem_id", "n_examples", "n_tests"), on="problem_id", validate="m:1")
        .with_columns(both=pl.col("x86").is_not_null() & pl.col("arm").is_not_null())
        .with_columns(
            passed=pl.col("both") & (pl.col("on_examples") == pl.lit("P").repeat_by(pl.col("n_examples")).list.join(""))
            & (pl.col("on_tests") == pl.lit("P").repeat_by(pl.col("n_tests")).list.join("")),
            matched=pl.col("both") & pl.col("on_examples").str.contains("^[PSK]+$")
            & (pl.col("on_examples").str.len_chars() == pl.col("n_examples"))
            & pl.col("on_tests").str.contains("^[PSK]+$") & (pl.col("on_tests").str.len_chars() == pl.col("n_tests")),
        )
        .sort("problem_id", "rank")
    )
    if set(built["problem_id"]) != set(problems["problem_id"]):
        raise ValueError(f"{options.built} and {options.out} hold different problems: rerun the build")

    summary = built.group_by("problem_id", maintain_order=True).agg(
        built=pl.col("both").sum(),
        passed=pl.col("passed").sum(),
        matched=pl.col("matched").sum(),
        candidates=pl.struct("solution_id", "arm").filter(pl.col("passed") & (pl.col("cpu") <= options.max_cpu)),
        outcomes=pl.concat_str("on_examples", "on_tests"),
        sources=pl.col("c"),
    ).join(problems.select("problem_id", "split", "group", "rating", "examples", "tests", "n_examples", "n_tests"),
           on="problem_id", validate="1:1")
    agreement = [count_agreement(outcomes, codes, examples + tests) for outcomes, codes, examples, tests
                 in summary.select("outcomes", "sources", "n_examples", "n_tests").iter_rows()]
    with ProcessPoolExecutor(options.workers) as pool:
        host = list(pool.map(check_on_host,
                             [[(row["solution_id"], row["arm"]) for row in rows] for rows in summary["candidates"]],
                             summary["examples"].to_list(), summary["tests"].to_list(), repeat(options.max_cpu)))
    summary = summary.with_columns(
        reference=pl.Series([solution_id for solution_id, _ in host], dtype=pl.String),
        cpu_host=pl.Series([cpu for _, cpu in host], dtype=pl.Float64),
        finishing=pl.Series([finishing for finishing, _ in agreement], dtype=pl.Int64),
        dissenting=pl.Series([dissenting for _, dissenting in agreement], dtype=pl.Int64),
    ).with_columns(
        task_id=pl.format("codeforces/{}/{}", "problem_id", "reference"),
        agreement_checked=pl.col("finishing") >= 3,
        multi_answer=(pl.col("dissenting") >= 2) & (pl.col("dissenting") >= options.dissent * pl.col("finishing")),
    ).join(built.select("problem_id", reference="solution_id", x86="x86", arm_ref="arm", cpu_vm="cpu"),
           on=["problem_id", "reference"], how="left", validate="1:1")

    decontam_items = pl.read_parquet(options.decontam_items)["item_id"]
    if missing := sorted(set(summary["task_id"].drop_nulls()) - set(decontam_items)):
        raise ValueError(f"{len(missing)} references are not in the decontamination run's items: {missing[:10]}")
    contaminated = pl.read_parquet(options.decontam).filter("drop")["item_id"].unique()
    prompt = pl.format(settings.prompt.template, "x86")
    codenet_prompts = pl.concat([pl.read_ndjson(path).select(content=pl.col("prompt").list.first().struct.field("content"))
                                 for path in settings.codenet.tasks.values()])["content"].unique()
    summary = summary.with_columns(reason=pl.when(pl.col("built") == 0).then(pl.lit("no_build"))
                                   .when((pl.col("passed") == 0) & (pl.col("matched") > 0)).then(pl.lit("layout_only"))
                                   .when(pl.col("passed") == 0).then(pl.lit("reference_fails"))
                                   .when(pl.col("candidates").list.len() == 0).then(pl.lit("slow"))
                                   .when(pl.col("reference").is_null()).then(pl.lit("host_fails"))
                                   .when(pl.col("multi_answer")).then(pl.lit("multi_answer"))
                                   .when(pl.col("task_id").is_in(contaminated.implode())).then(pl.lit("contaminated"))
                                   .when(prompt.is_in(codenet_prompts.implode())).then(pl.lit("shared_x86")))
    shared = (pl.col("reason").is_null() & prompt.is_duplicated().over(pl.col("reason").is_null()))
    summary = summary.with_columns(reason=pl.when(shared).then(pl.lit("shared_x86")).otherwise(pl.col("reason"))
                                   .fill_null("kept"))

    tokenizer = Tokenizer.from_file(hf_hub_download(settings.rl.tokenizer, "tokenizer.json",
                                                    revision=settings.rl.tokenizer_revision))
    kept = summary.filter(reason="kept").with_columns(prompt=pl.concat_list(pl.struct(role=pl.lit("user"), content=prompt)))
    tokens = [len(prompt_encoding.ids) + len(reference_encoding.ids) for prompt_encoding, reference_encoding in zip(
        tokenizer.encode_batch(kept["prompt"].list.first().struct.field("content").to_list()),
        tokenizer.encode_batch(kept["arm_ref"].to_list()), strict=True)]
    kept = kept.with_columns(tokens=pl.Series(tokens, dtype=pl.Int64)).with_columns(over_length=pl.col("tokens") > context)
    for split, path in options.tasks.items():
        frame = kept.filter(split=split).select("task_id", "arm_ref", tests="examples", prompt="prompt").sort("task_id")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_ndjson(path)
        log.info("wrote", path=str(path), tasks=frame.height)
    options.eval.parent.mkdir(parents=True, exist_ok=True)
    kept.select("problem_id", "split", "tests").sort("problem_id").write_parquet(options.eval)
    drops = summary.join(kept.select("problem_id", "tokens", "over_length"), on="problem_id", how="left", validate="1:1")
    drops.select("problem_id", "split", "group", "rating", "reason", "reference", "built", "passed", "matched", "finishing",
                 "dissenting", "agreement_checked", "cpu_vm", "cpu_host", "tokens", "over_length",
                 ).sort("problem_id").write_parquet(options.drops)
    log.info("codeforces", eval=str(options.eval), drops=str(options.drops),
             reasons=drops.group_by("split", "reason").len().sort("split", "reason").to_dicts(),
             kept_unchecked_agreement=kept.filter(~pl.col("agreement_checked")).height,
             over_length=kept.group_by("split").agg(pl.col("over_length").sum()).sort("split").to_dicts())


if __name__ == "__main__":   # pool workers import this module
    main()
