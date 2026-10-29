import json, sys, re, collections, hashlib
def load(p):
    out=[]
    for l in open(p):
        j=json.loads(l); out.append(j["text"] if "text" in j else j)
    return out
tr, va, te = (load(p) for p in sys.argv[1:4])
print("sizes", len(tr), len(va), len(te))
print("train keys:", list(tr[0].keys())[:30])
norm = lambda s: re.sub(r"\s+", "", s or "")
def keyset(rows): return collections.Counter(norm(r["func_def"]) for r in rows)
ktr, kva, kte = keyset(tr), keyset(va), keyset(te)
inter = lambda a,b: sum(1 for k in b if k in a)
print("test_real funcs (whitespace-normalized) also in train_real_simple_io:", sum(c for k,c in kte.items() if k in ktr), "/", len(te))
print("valid_real funcs also in train_real_simple_io:", sum(c for k,c in kva.items() if k in ktr), "/", len(va))
print("test_real funcs also in valid_real:", sum(c for k,c in kte.items() if k in kva))
print("dup rows within train_real_simple_io:", len(tr)-len(ktr))
# same fname+path overlap
p = lambda rows: set((r["path"], r["fname"]) for r in rows)
print("test (path,fname) in train:", len(p(te)&p(tr)))
repo = lambda rows: set("/".join(r["path"].split("/")[:2]) for r in rows)
print("test repos also in train_real_simple_io repos:", len(repo(te)&repo(tr)), "/", len(repo(te)))
# asm targets in main branch
a = te[0]["asm"]
print("main asm targets:", a.get("target") if isinstance(a, dict) and "target" in a else (list(a.keys()) if isinstance(a, dict) else a[:1]))
# trivial flags in train
print("train trivial flag:", collections.Counter(str(r.get("real_io_pairs_are_trivial")) for r in tr))
