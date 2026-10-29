import json, sys, re, collections
angha = set(open(sys.argv[3]).read().split())
norm = lambda s: re.sub(r"\s+", "", s or "")
U=lambda j: j.get("text", j)
test = set(norm(U(json.loads(l))["func_def"]) for l in open(sys.argv[2]))
n=0; same=0; noout=0; triv=collections.Counter(); intest=0; arepo=0; nio=collections.Counter(); keys=set(); static=0
first=None; repos=set(); ahits=collections.Counter()
for l in open(sys.argv[1]):
    r=U(json.loads(l)); n+=1
    if first is None: first=r
    io=r.get("angha_io_pairs") or []
    nio[len(io)]+=1
    outs=[json.dumps(x["output"],sort_keys=True) for x in io]
    if outs and len(set(outs))==1: same+=1
    if io and all(not x["output"] for x in io): noout+=1
    triv[str(r.get("angha_io_pairs_are_trivial"))]+=1
    k=norm(r["func_def"]); keys.add(k)
    if k in test: intest+=1
    parts=r["path"].split("/"); repos.add(tuple(parts[:2]))
    if len(parts)>1 and parts[1] in angha: arepo+=1; ahits[parts[1]]+=1
    if r["func_def"].lstrip().startswith("static"): static+=1
print("n",n,"distinct func_def",len(keys),"nio",nio.most_common(3))
print("trivial flag",triv, "all-outputs-identical",f"{same/n:.1%}","empty-output",f"{noout/n:.1%}")
print("in test_real (exact, ws-normalized):",intest)
print("distinct repos",len(repos),"rows from repos named like public-AnghaBench repos:",arepo, f"{arepo/n:.1%}", ahits.most_common(10))
print("static", f"{static/n:.1%}")
print("=== example func_def ===\n", first["func_def"][:600])
print("=== example angha_deps ===\n", (first.get("angha_deps") or "")[:1500])
print("=== example io ===\n", json.dumps(first["angha_io_pairs"][:2])[:800])
print("=== wrapper head ===\n", (first.get("angha_exe_wrapper") or "")[:300])
