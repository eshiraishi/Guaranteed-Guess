"""Compile-only portability probe of ExeBench rows on macOS clang. No execution of dataset code."""
import json, sys, re, os, subprocess, tempfile, random, collections
from concurrent.futures import ThreadPoolExecutor
src, kind, nmax, wrap_n, inc, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
U = lambda j: j.get("text", j)
rows = []
with open(src) as f:
    for i, l in enumerate(f):
        rows.append(U(json.loads(l)))
        if len(rows) >= nmax: break
dep_key = "real_deps" if kind == "real" else "angha_deps"
wr_key = "real_exe_wrapper" if kind == "real" else "angha_exe_wrapper"
work = tempfile.mkdtemp(prefix="ebct_", dir=out)
def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stderr
def one(idx):
    r = rows[idx]; d = os.path.join(work, str(idx)); os.makedirs(d)
    deps = r.get(dep_key) or ""
    if kind == "synth":
        deps = deps.replace("typedef int bool;", "")
        dfs = (r.get("angha_io_pairs") or [{}])[0].get("dummy_funcs") or ""
        wdeps_extra = "\n" + dfs
    else:
        wdeps_extra = ""
    open(f"{d}/f.c", "w").write(deps + "\n" + r["func_def"] + "\n")
    res = {}
    rc, err = run(["clang", "-w", "-O2", "-S", "-arch", "arm64", "f.c", "-o", "arm.s"], d)
    res["arm_ok"] = rc == 0
    res["arm_err"] = "" if rc == 0 else (err.strip().splitlines() or [""])[0][:160]
    rc2, _ = run(["clang", "-w", "-O2", "-S", "-target", "x86_64-apple-darwin", "f.c", "-o", "x86.s"], d)
    res["x86_ok"] = rc2 == 0
    if res["arm_ok"]:
        s = open(f"{d}/arm.s").read()
        res["sym_present"] = bool(re.search(rf"^_{re.escape(r['fname'])}:", s, re.M))
        res["global"] = bool(re.search(rf"\.globl\s+_{re.escape(r['fname'])}\b", s))
    if idx < wrap_n and res["arm_ok"] and r.get(wr_key):
        # mimic exebench._DefaultAssembler: deps + extern decl included in extern "C" block; function from .s
        sig = (r.get("func_head_types") or "").replace("extern", "")
        open(f"{d}/deps.c", "w").write(deps + wdeps_extra + f"\nextern {sig};\n")
        w = re.sub(r'extern\s\"C\"\s\{\s.*\s\}', 'extern "C" \n{\n#include "deps.c"\n}\n', r[wr_key])
        open(f"{d}/w.cpp", "w").write(w)
        # make function globally visible for linking: strip 'static'/'inline' then recompile
        fd = re.sub(r"\b(static|inline|__inline__|__inline)\b\s*", "", r["func_def"].split("{",1)[0], count=0) + "{" + r["func_def"].split("{",1)[1]
        open(f"{d}/g.c", "w").write(deps + "\n" + fd + "\n")
        rcg, _ = run(["clang", "-w", "-O2", "-S", "-arch", "arm64", "-fcommon", "g.c", "-o", "g.s"], d)
        rcw, errw = run(["clang++", "-w", "-std=c++17", "-O0", "-arch", "arm64", "-I", inc, "w.cpp", "g.s", "-o", "w.x"], d)
        res["wrap_ok"] = (rcg == 0 and rcw == 0)
        if not res["wrap_ok"]:
            m = [l for l in errw.splitlines() if "error" in l]
            res["wrap_err"] = (m[0] if m else errw[:160])[:200]
        # empty .s must fail to link
        open(f"{d}/empty.s", "w").write("")
        rce, _ = run(["clang++", "-w", "-std=c++17", "-O0", "-arch", "arm64", "-I", inc, "w.cpp", "empty.s", "-o", "e.x"], d)
        res["empty_links"] = rce == 0
    return res
with ThreadPoolExecutor(10) as ex:
    results = list(ex.map(one, range(len(rows))))
n = len(results)
c = collections.Counter()
for r in results:
    for k, v in r.items():
        if isinstance(v, bool) and v: c[k] += 1
nw = sum(1 for r in results if "wrap_ok" in r)
print(f"{os.path.basename(src)} n={n}: arm_ok={c['arm_ok']/n:.1%} x86_ok={c['x86_ok']/n:.1%} "
      f"sym_present_at_O2(of arm_ok)={c['sym_present']/max(c['arm_ok'],1):.1%} global={c['global']/max(c['arm_ok'],1):.1%} "
      f"| wrapper tried={nw} wrap_link_ok={c['wrap_ok']}/{nw} empty_s_links={c['empty_links']}/{nw}")
print("top arm errors:", collections.Counter(re.sub(r"^.*?error: ", "", r["arm_err"]) for r in results if not r["arm_ok"]).most_common(8))
print("top wrapper errors:", collections.Counter(re.sub(r"^.*?error: ", "", r.get("wrap_err","")) for r in results if r.get("wrap_ok") is False).most_common(8))
