import json, sys, collections
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
r = rows[0]
print("text:", str(r["text"])[:300])
m = r["text"]
print("meta keys:", list(m.keys()))
for k in ["path","fname","func_def","func_head_types","signature","real_deps","real_iospec","ref"]:
    v = m.get(k); s = v if isinstance(v,str) else json.dumps(v)
    print(f"=== {k} ===\n{(s or '')[:1500]}\n")
asm = m.get("asm")
print("asm:", type(asm), list(asm.keys()) if isinstance(asm, dict) else "")
if isinstance(asm, dict):
    for t,v in list(asm.items()):
        print(" target", t, "->", (json.dumps(v)[:150]))
io = m.get("real_io_pairs")
print("io type", type(io), len(io) if io else None)
print(json.dumps(io)[:1500])
print("=== real_exe_wrapper ===")
print((m.get("real_exe_wrapper") or "")[:4500])
c = collections.Counter(len(x["text"].get("real_io_pairs") or []) for x in rows)
print("io count dist", c.most_common(10))
