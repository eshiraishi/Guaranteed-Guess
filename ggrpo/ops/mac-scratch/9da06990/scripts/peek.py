import json, sys, collections
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
r = rows[0]
print("keys:", list(r.keys()))
def show(k, n=600):
    v = r.get(k)
    s = json.dumps(v) if not isinstance(v, str) else v
    print(f"=== {k} ===\n{s[:n]}\n")
for k in ["path","fname","func_def","func_head_types","signature","real_deps","real_iospec","ref"]:
    show(k, 1200)
asm = r["asm"]
print("asm type:", type(asm))
if isinstance(asm, dict):
    print("targets:", asm.get("target"))
elif isinstance(asm, list):
    print("targets:", [a.get("target") for a in asm])
io = r["real_io_pairs"]
print("io type", type(io), (len(io) if io else None))
print(json.dumps(io)[:1500])
print("=== real_exe_wrapper ===")
print((r["real_exe_wrapper"] or "")[:4000])
# stats
n_io = collections.Counter()
for x in rows:
    io = x.get("real_io_pairs")
    n_io[len(io) if isinstance(io, list) else (len(io.get("input",[])) if isinstance(io, dict) else 0)] += 1
print("io count dist", n_io.most_common(10))
