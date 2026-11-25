"""Build the one W&B report that groups every run by the comparison it belongs to (Enzo, Nov 18: too many runs,
nowhere to look).

The project workspace shows all 14 runs on every chart, which mixes arms that are not comparable: the 200k-era GRPO
variants start from a different SFT model than the current run, and SFT and RL share almost no metric keys. A report
fixes that by giving each section its own run set, filtered to exactly the run ids of one comparison, so a chart in a
section can only ever show runs that belong together.

Titles and axes are written for a reader who does not know the metric keys, and each chart carries one scale only: a
share and a token count on one axis would make both unreadable. Superseded runs (the October variants and the 200k
pilots) are left out entirely (Enzo, Nov 18) -- they stay in the project workspace for provenance.

Re-running this rewrites the report named by [track].report, so the link never changes. report.save() always creates a
new report, so without that pinned url every run would publish another copy under the same title.
"""
import os

import wandb_workspaces.reports.v2 as wr

from ggrpo.settings import Settings

TITLE = "GG RLVR - one dashboard"

VERDICTS = {"train/verdict/pass": "passes the tests", "train/verdict/wrong_output": "wrong answer",
            "train/verdict/assemble_err": "will not assemble", "train/verdict/crash": "crashes",
            "train/verdict/timeout": "too slow"}

SECTIONS = [
    ("Does RL beat the model it started from?",
     ["gg", "sft-full-seed0-evals", "sft-full-seed0", "grpo-codenet-seed0"],
     [("bar", {"eval/codenet_val_beam8": "best guess", "eval/codenet_val_sample8": "8 tries"},
       "Share of the 383 validation problems solved", "share solved")]),

    ("Is the run learning?",
     ["grpo-codenet-seed0"],
     [("line", {"train/rewards/reward_pass/mean": "share passing"},
       "Share of attempts that pass the tests", "share passing"),
      ("line", VERDICTS, "What the attempts do", "share of attempts"),
      ("line", {"train/completions/mean_length": "average answer"}, "How long the answers are", "tokens"),
      ("line", {"train/completions/clipped_ratio": "cut off"}, "Answers that hit the length limit", "share cut off"),
      ("line", {"train/entropy": "entropy"}, "How varied the answers are", "entropy"),
      ("line", {"train/grad_norm": "update size"}, "How big each weight update is", "gradient norm"),
      ("line", {"train/step_time": "seconds"}, "Time per training step", "seconds")]),

    ("Is the reward real, or would any reward do?",
     ["grpo-codenet-seed0", "grpo-codenet-random-seed0"],
     [("line", {"train/rewards/reward_pass/mean": "share passing"},
       "Share passing: real reward vs shuffled reward", "share passing"),
      ("bar", {"eval/codenet_val_beam8": "best guess"},
       "Validation problems solved: real vs shuffled reward", "share solved")]),

    ("The imitation side (plain fine-tuning)",
     ["sft-full-seed0", "sft-redo-200k-v2-seed0", "sft-redo-200k-seed0", "sft-seed0"],
     [("line", {"train/loss": "training"}, "Error while training", "loss"),
      ("line", {"eval/loss": "held-out"}, "Error on held-out data", "loss"),
      ("line", {"eval/mean_token_accuracy": "accuracy"}, "Tokens predicted correctly", "share correct")]),
]


def build_panel(kind: str, series: dict[str, str], title: str, axis: str) -> wr.BarPlot | wr.LinePlot:
    """One panel, one scale. A bar plot reads each run's last value of the key; a line plot reads it against the step.

    line_titles renames each series in the legend, so a reader never has to know the metric key behind it.
    """
    if kind == "bar":
        return wr.BarPlot(title=title, metrics=list(series), title_x=axis, line_titles=series,
                          layout=wr.Layout(w=12, h=8))
    return wr.LinePlot(title=title, x="train/global_step", y=list(series), title_x="training step", title_y=axis,
                       line_titles=series, layout=wr.Layout(w=12, h=8))


def build_section(settings: Settings, title: str, run_ids: list[str], panels: list) -> list:
    """A heading and a panel grid whose single run set can only serve this comparison's runs."""
    runset = wr.Runset(entity=settings.track.entity, project=settings.track.project, name=title,
                       filters=f"ID in {run_ids}")
    return [wr.H2(title),
            wr.PanelGrid(runsets=[runset], panels=[build_panel(*panel) for panel in panels])]


def publish(settings: Settings) -> str:
    """Write the report and return its url."""
    # the reports library builds its own wandb.Api() with no argument, so the key has to be in the environment
    os.environ["WANDB_API_KEY"] = settings.wandb_api_key.get_secret_value()
    blocks = [wr.P("Validation only. The test problems, HumanEval-C and BringUp are scored once, at the end.")]
    for section in SECTIONS:
        blocks += build_section(settings, *section)
    description = "One section per comparison, validation only."
    if settings.track.report:
        report = wr.Report.from_url(settings.track.report)
        report.title, report.description, report.width, report.blocks = TITLE, description, "fluid", blocks
    else:
        report = wr.Report(entity=settings.track.entity, project=settings.track.project, title=TITLE,
                           description=description, width="fluid", blocks=blocks)
    report.save()
    return report.url


if __name__ == "__main__":
    print(publish(Settings()))
