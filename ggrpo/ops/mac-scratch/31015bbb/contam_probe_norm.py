"""Feasibility probe: exact and 10-instruction-window overlap between benchmark x86 and one GG SFT shard."""
import re
import sys
import time

import polars as pl

K = 10
SYM = re.compile(r"\b_[A-Za-z_][A-Za-z0-9_.$]*")


def normalize(asm: str) -> list[str]:
    out = []
    for line in asm.splitlines():
        line = line.split("##")[0].rstrip()
        s = line.strip()
        if not s or (s.startswith(".") and not s.startswith(".L")):
            continue
        out.append(SYM.sub("_S", s))
    return out


def windows(lines: list[str]) -> set[int]:
    return {hash("\n".join(lines[i : i + K])) for i in range(max(1, len(lines) - K + 1))}

