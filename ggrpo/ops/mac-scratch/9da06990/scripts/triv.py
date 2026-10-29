import json, sys, collections
for p in sys.argv[1:]:
    rows=[json.loads(l) for l in open(p)]
    rows=[r.get("text", r) for r in rows]
    n=len(rows); same=0; noout=0; outeqin=0; nio=collections.Counter(); void=0
    for r in rows:
        io=r["real_io_pairs"] or []
        nio[len(io)]+=1
        outs=[json.dumps(x["output"],sort_keys=True) for x in io]
        if outs and len(set(outs))==1: same+=1
        if all(not x["output"] for x in io): noout+=1
        if io and all(all((x["input"] or {}).get(k)==v for k,v in (x["output"] or {}).items()) for x in io): outeqin+=1
        if r["signature"][0].strip()=="void": void+=1
    print(p.split("/")[-1], "n",n, "nio",nio.most_common(3), "all-outputs-identical",f"{same/n:.1%}", "empty-output",f"{noout/n:.1%}", "outputs==inputs(no effect)",f"{outeqin/n:.1%}", "void",f"{void/n:.1%}")
rows=[json.loads(l) for l in open(sys.argv[1])]
for r in rows[:3]:
    print("----", r["func_def"][:300]); print(json.dumps(r["real_io_pairs"][:2])[:400])
