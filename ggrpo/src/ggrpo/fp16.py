"""The RL starting model as TRL's colocated vLLM should load it: in fp16 (training host, before `trl grpo`).
TRL 1.16 starts vLLM with `LLM(model=model.name_or_path)` and no dtype, so vLLM takes the dtype from config.json. In bf16
the trainer's and vLLM's token log-probs drift apart with length, and the importance-sampling correction loses up to half
the weight of ~8k-token completions; fp16 on both sides removes most of the gap (pilot, Nov 10, 2026: mean ratio
0.45-0.49 in bf16, 1.01-1.02 in fp16, same step time; Qi et al., arXiv 2510.26788). fp16.out links every file of
fp16.model except config.json, whose dtype becomes float16. The trainer does not read the dtype from it: it loads fp32
master weights (`dtype: float32`) and runs fp16 autocast (`fp16: true`)."""
import json

import structlog
from pydantic_settings import CliApp

from ggrpo.settings import Settings

settings = CliApp.run(Settings)
log = structlog.get_logger("ggrpo")
config = json.loads((settings.fp16.model / "config.json").read_text())
dtype_keys = [key for key in ("torch_dtype", "dtype") if key in config]
if not dtype_keys:
    raise SystemExit(f"{settings.fp16.model / 'config.json'} names no dtype")
settings.fp16.out.mkdir(parents=True)
for path in settings.fp16.model.iterdir():
    if path.name != "config.json":
        (settings.fp16.out / path.name).symlink_to(path.resolve())
(settings.fp16.out / "config.json").write_text(json.dumps(config | dict.fromkeys(dtype_keys, "float16"), indent=2))
log.info("fp16 view", model=str(settings.fp16.model), out=str(settings.fp16.out), was={key: config[key] for key in dtype_keys})
