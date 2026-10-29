"""Write the report's numbers and figures from the scored evals in config.toml [track] and the [latex] training logs.

`uv run python -m ggrpo.latex`, or ops/report.sh, which also builds the PDF. Every number becomes
\\resdef{run/step/metric/stat}{value}, with step `final` for each run's last evaluated step. Scored files that do not
exist yet are skipped, and the report prints those values as pending.
"""
import json
from datetime import UTC, datetime
from math import comb

import polars as pl
import structlog

from ggrpo.settings import Settings
from ggrpo.track import read_tasks

settings = Settings()
log = structlog.get_logger("ggrpo")
decontam = json.loads(settings.latex.decontam.read_text())
twins = decontam["mbpp_humaneval_twins"]   # MBPP task -> HumanEval index of the same problem
clean_excluded = {   # benchmark -> tasks left out of its clean subset
    "humaneval": {f"humaneval/{index}" for index in twins.values()},
    "bringup": set(decontam["bringup_contaminated"]["programs"]),
}
baseline = settings.track.runs[settings.latex.baseline].evals[0]

numbers = {
    "data/twins": len(twins),
    "data/twin_problems": len(clean_excluded["humaneval"]),
    "data/twins_train": pl.read_ndjson(settings.data.train)["task_id"].is_in(list(twins)).sum(),
    "data/bringup_contaminated": len(clean_excluded["bringup"]),
}
dev = []   # run, step, beam and sampled pass@1 (%): the dev figure
# A run enters the report only when every file registered at its final step is in (Enzo, Oct 23)
complete = {run_id for run_id, run in settings.track.runs.items() if all(path.exists() for path in run.evals[max(run.evals)].values())}
numbers |= {f"{run_id}/complete": 1 for run_id in complete}
for run_id, run in settings.track.runs.items():
    for step, files in sorted(run.evals.items()):
        # `final` is the last registered checkpoint, so register a run's final step before its files land
        keys = [f"{run_id}/{step}"] + ([f"{run_id}/final"] if step == max(run.evals) else [])
        for metric, path in files.items():
            if not path.exists():
                continue
            tasks = read_tasks(path)
            clean = tasks.filter(~pl.col("task_id").is_in(clean_excluded.get(metric.split("_")[0], set())))
            stats = {
                "n": tasks.height,
                "solved": tasks["passed"].gt(0).sum(),   # sample8: solved in any of 8
                "pass1": f"{100 * tasks['passed'].mean():.1f}",
                "clean_n": clean.height,
                "clean_solved": clean["passed"].gt(0).sum(),
                "clean_pass1": f"{100 * clean['passed'].mean():.1f}",
            }
            if metric in baseline and not metric.endswith("sample8"):   # flips need one prediction per task
                flips = tasks.join(read_tasks(baseline[metric]).rename({"passed": "base"}), on="task_id", validate="1:1")
                gained = flips.filter(pl.col("passed").gt(0) & pl.col("base").eq(0))
                up, down = gained.height, flips.filter(pl.col("passed").eq(0) & pl.col("base").gt(0)).height
                p = min(1.0, 2 * sum(comb(up + down, k) for k in range(min(up, down) + 1)) / 2 ** (up + down))
                stats |= {"up": up, "down": down, "p": f"{p:.2g}"}   # exact two-sided sign test
                if (sample8 := metric.split("_")[0] + "_sample8") in baseline:   # gained problems the base solves in some of 8 samples
                    stats["upsampled"] = gained.join(read_tasks(baseline[sample8]), on="task_id", suffix="8")["passed8"].gt(0).sum()
            if metric.endswith("sample8") and "verdict" in (verdicts := pl.read_ndjson(path)).columns:   # samples that fail to assemble
                stats |= {"samples": verdicts.height, "asmerr": verdicts.filter(verdict="assemble_err").height}
            numbers |= {f"{key}/{metric}/{stat}": value for key in keys for stat, value in stats.items()}
        point = [numbers.get(f"{run_id}/{step}/{metric}/pass1") for metric in ("dev_beam8", "dev_sample8")]
        if any(point):   # beam and sampled land at different times: plot each as soon as it exists
            dev.append((run_id, step, *(float(value) if value else None for value in point)))

histories = {
    run_id: pl.from_dicts(json.loads(path.read_text())["log_history"], infer_schema_length=None)
    for run_id, path in settings.latex.trainer_states.items()
    if path.exists()
}
training = {   # GRPO runs only: SFT logs a loss, not a reward
    run_id: history.filter(pl.col("reward").is_not_null())
    .with_columns(data_pass=pl.col("epoch").ceil().cast(pl.Int64))   # 1 for steps 1-53, 2 for 54-106
    for run_id, history in histories.items()
    if "reward" in history.columns and run_id in complete
}
for run_id, history in training.items():
    passes = [(f"e{n}", history.filter(data_pass=n)) for n in history["data_pass"].unique().sort()]
    for name, part in [("all", history), *passes]:
        numbers |= {
            f"{run_id}/train/{name}/reward": f"{100 * part['reward'].mean():.1f}",
            f"{run_id}/train/{name}/zerostd": f"{100 * part['frac_reward_zero_std'].mean():.1f}",
            f"{run_id}/train/{name}/entropy": f"{part['entropy'].mean():.3g}",
            f"{run_id}/train/{name}/length": f"{part['completions/mean_length'].mean():.0f}",
            f"{run_id}/train/{name}/clipped": f"{100 * part['completions/clipped_ratio'].mean():.1f}",
            f"{run_id}/train/{name}/steptime": f"{part['step_time'].mean():.0f}",
        }
    numbers[f"{run_id}/train/steps"] = history["step"].max()
    numbers[f"{run_id}/train/hours"] = f"{history['step_time'].sum() / 3600:.1f}"

for run_id, history in histories.items():
    if "mean_token_accuracy" in history.columns:   # SFT: share of reference tokens predicted at its first and last update
        accuracy, loss = history["mean_token_accuracy"].drop_nulls(), history["loss"].drop_nulls()
        numbers |= {f"{run_id}/train/first/loss": f"{loss[0]:.3f}", f"{run_id}/train/last/loss": f"{loss[-1]:.3f}"}
        # the last point of the SFT figure, which averages over 10 steps
        numbers |= {f"{run_id}/train/last10/loss": f"{loss.tail(10).mean():.3f}",
                    f"{run_id}/train/last10/accuracy": f"{100 * accuracy.tail(10).mean():.1f}"}
        numbers |= {f"{run_id}/train/first/accuracy": f"{100 * accuracy[0]:.1f}",
                    f"{run_id}/train/last/accuracy": f"{100 * accuracy[-1]:.1f}"}

settings.latex.out.mkdir(parents=True, exist_ok=True)
(settings.latex.out / "numbers.tex").write_text(
    f"% Written by python -m ggrpo.latex at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Do not edit.\n"
    + "".join(f"\\resdef{{{key}}}{{{value}}}\n" for key, value in numbers.items())
)

# Figures as pgfplots code with the data inline, so the report also works as one pasted .tex file
trained = [run_id for run_id in settings.track.runs if run_id != settings.latex.baseline]
colors = dict(zip(trained, [f"series{n}" for n in range(1, 6)]))   # one color per run in every figure (macros.tex)
axis = (r"width=0.48\linewidth, height=4.2cm, xlabel=Step, enlarge x limits=0.03, tick pos=left, tick label style={font=\scriptsize}, "
        r"label style={font=\small}, legend style={font=\scriptsize, cells={anchor=west}}")
panels = []
for column, ylabel in [("reward", "Passed (\\%)"), ("frac_reward_zero_std", "Groups with no gradient (\\%)")]:
    plots = "".join(
        f"\\addplot[{colors[run_id]}, thick] coordinates {{"
        + " ".join(f"({step},{100 * value:.1f})" for step, value in smooth.select("step", column).iter_rows())
        + "};\n"
        for run_id, smooth in ((run_id, history.select("step", pl.col(column).rolling_mean(10, min_samples=1)))
                               for run_id, history in training.items())
    )
    panels.append(f"\\begin{{tikzpicture}}\\begin{{axis}}[{axis}, ylabel={{{ylabel}}}]\n{plots}\\end{{axis}}\\end{{tikzpicture}}")
if len(training) > 1:   # one line needs no legend: the caption names it
    legend = ",".join(f"{{{settings.latex.labels.get(run_id, run_id)}}}" for run_id in training)   # braced: labels have commas
    panels[0] = panels[0].replace("\\end{axis}", f"\\legend{{{legend}}}\n\\end{{axis}}").replace("]\n", ", legend pos=south east]\n", 1)
(settings.latex.out / "training.tex").write_text("\\hfill".join(panels) + "\n")

# SFT logs a loss and the share of reference tokens it predicts instead of a reward, so it gets its own figure
sft = {run_id: history.filter(pl.col("loss").is_not_null()) for run_id, history in histories.items()
       if "mean_token_accuracy" in history.columns and run_id in complete}
panels = []
fixed = r", yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1}"   # 99.0, not 99, next to 99.2
for column, ylabel, scale, ticks in [("loss", "Loss ($\\times 10^{-2}$)", 100, ""),
                                     ("mean_token_accuracy", "Token accuracy (\\%)", 100, fixed)]:
    plots = "".join(
        f"\\addplot[{colors[run_id]}, thick] coordinates {{"
        + " ".join(f"({step},{scale * value:.4g})"
                   for step, value in history.select("step", pl.col(column).rolling_mean(10, min_samples=1)).iter_rows())
        + "};\n"
        for run_id, history in sft.items()
    )
    panels.append(f"\\begin{{tikzpicture}}\\begin{{axis}}[{axis}, ylabel={{{ylabel}}}{ticks}]\n{plots}\\end{{axis}}\\end{{tikzpicture}}")
(settings.latex.out / "sft_training.tex").write_text("\\hfill".join(panels) + "\n")

# Beam and sampled dev on one percent scale, padded by at least 10 points so a change of a few problems is not magnified
curves = pl.DataFrame(dev, schema=["run", "step", "beam", "sampled"], orient="row")
values = pl.concat([curves["beam"], curves["sampled"]]).drop_nulls()
ymin, ymax = max(0, 10 * (values.min() // 10) - 10), min(100, 10 * -(-values.max() // 10) + 10)
base = curves.filter(run=settings.latex.baseline, step=0)
plots = []
runs = curves.filter(pl.col("run") != settings.latex.baseline).group_by("run", maintain_order=True)
finished = [(run_id, curve) for (run_id,), curve in runs if run_id in complete]
for run_id, curve in finished:   # a run still training is left out until its final checkpoint is evaluated
    curve = pl.concat([base, curve]).sort("step")
    label = settings.latex.labels.get(run_id, run_id)
    # one legend entry per line: color names the method, line style the decoding
    for column, style, decoding in [("beam", "mark=*", "beam search"), ("sampled", "mark=square*, dashed", "sampling")]:
        series = curve.select("step", column).drop_nulls()
        if series.height < 2:   # only the base point: no checkpoint of this run has the metric yet
            continue
        points = " ".join(f"({step},{value:.1f})" for step, value in series.iter_rows())
        plots.append(f"\\addplot[{colors[run_id]}, {style}] coordinates {{{points}}};\n\\addlegendentry{{{label} ({decoding})}}\n")
(settings.latex.out / "dev.tex").write_text(
    f"\\begin{{tikzpicture}}\\begin{{axis}}[{axis.replace('0.48', '0.5')}, ylabel={{Solved (\\%)}}, ymin={ymin:.0f}, "
    f"ymax={ymax:.0f}, legend pos=outer north east]\n"
    + "".join(plots)
    + "\\end{axis}\\end{tikzpicture}\n"
)

# The GG paper's two benchmarks: GG as reported, then each model whose final HumanEval-C is in (a run still training is left out)
benches = {"humaneval_beam8": "HumanEval-C", "bringup_beam8": "BringUp"}
models = [("GG (reported) \\citep{gg}", "draw=black!60, pattern=north east lines, pattern color=black!60", settings.latex.reported)]   # hatched: not our run
for run_id, key in [(settings.latex.baseline, f"{settings.latex.baseline}/0"), *((run_id, f"{run_id}/final") for run_id in trained)]:
    measured = {metric: (numbers[f"{key}/{metric}/solved"], numbers[f"{key}/{metric}/n"])
                for metric in benches if f"{key}/{metric}/solved" in numbers}
    if run_id in complete:
        color = colors.get(run_id, "ggours")
        models.append((settings.latex.labels.get(run_id, run_id), f"fill={color}, draw={color}", measured))
bars = "".join(
    f"\\addplot[{style}] coordinates {{"
    + " ".join(f"({benches[metric]},{100 * solved / n:.1f}) [{solved}]" for metric, (solved, n) in measured.items())
    + "};\n"
    for _, style, measured in models
)
top = max(100 * solved / n for _, _, measured in models for solved, n in measured.values())
(settings.latex.out / "benchmarks.tex").write_text(
    f"\\begin{{tikzpicture}}\\begin{{axis}}[ybar, bar width=8pt, width=0.55\\linewidth, height=4.5cm, ymin=0, "
    f"ymax={10 * -(-top // 10) + 10:.0f}, ylabel={{Solved (\\%)}}, symbolic x coords={{{','.join(benches.values())}}}, "
    r"xtick=data, tick pos=left, enlarge x limits=0.5, nodes near coords, point meta=explicit symbolic, "
    r"every node near coord/.append style={font=\tiny}, tick label style={font=\scriptsize}, label style={font=\small}, "
    r"legend style={font=\scriptsize, cells={anchor=west}}, legend pos=outer north east, area legend]" "\n"
    + bars + f"\\legend{{{','.join(f'{{{label}}}' for label, _, _ in models)}}}\n\\end{{axis}}\\end{{tikzpicture}}\n"
)
log.info("latex.done", out=str(settings.latex.out), numbers=len(numbers), runs=list(training))
