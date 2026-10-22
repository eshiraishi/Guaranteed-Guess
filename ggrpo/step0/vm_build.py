"""Runs INSIDE the macOS 13 VM (stdlib only): compile x86/arm at one opt level, validate ref and empty prediction.
Usage: vm_build.py jobs.jsonl out.jsonl [workers]. Commands identical to rl/data/build_humaneval.py compile_asm."""
import json, os, subprocess, sys, tempfile
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_copy"))
from harness import Harness

def compile_asm(src, opt, target):
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "code.c"); s = os.path.join(d, "code.s")
        open(c, "w").write(src)
        cmd = (f"clang -S -target x86_64-apple-darwin {c} -o {s} -{opt}" if target == "x86"
               else f"clang -arch arm64 -S {c} -o {s} -{opt}")
        r = subprocess.run(cmd, shell=True, capture_output=True, timeout=20)
        return open(s).read() if r.returncode == 0 else None

def work(j):
    x86 = compile_asm(j["code"], j["opt"], "x86"); arm = compile_asm(j["code"], j["opt"], "arm")
    res = {"task_id": j["task_id"], "source": j["source"], "opt": j["opt"], "x86": x86, "arm_ref": arm,
           "support_src": j["support_src"], "cflags": j["cflags"], "ref_verdict": None, "neg_verdict": None,
           "ref_stderr": "", "drop": None}
    if x86 is None or arm is None: res["drop"] = "compile_fail"; return res
    if j["support_src"] is None: res["drop"] = "no_driver"; return res
    h = Harness(); task = {"support_src": j["support_src"], "cflags": j["cflags"], "ldflags": []}
    ref = h.check(arm, task); neg = h.check("", task)
    res.update(ref_verdict=ref.verdict.value, neg_verdict=neg.verdict.value, ref_stderr=ref.stderr_tail)
    return res

if __name__ == "__main__":
    jobs = [json.loads(l) for l in open(sys.argv[1])]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as p, open(sys.argv[2], "w") as f:
        for r in p.imap_unordered(work, jobs, chunksize=4): f.write(json.dumps(r) + "\n")
