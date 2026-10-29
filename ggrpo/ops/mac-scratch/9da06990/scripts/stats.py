import json, sys, collections, re
rows = [json.loads(l)["text"] for l in open(sys.argv[1])]
n = len(rows)
print("n", n)
triv = collections.Counter(str(r.get("real_io_pairs_are_trivial")) for r in rows)
print("real_io_pairs_are_trivial:", triv)
allsame = 0; ret_only=0
for r in rows:
    outs = [json.dumps(p["output"], sort_keys=True) for p in r["real_io_pairs"]]
    if len(set(outs)) == 1: allsame += 1
print("all 10 outputs identical:", allsame, f"{allsame/n:.1%}")
inc = collections.Counter()
linuxish = re.compile(r'#include\s*[<"]((linux|asm|bits|sys/(sysinfo|prctl|epoll|inotify|signalfd|eventfd|timerfd|io)|malloc|endian|byteswap|features|values|error|argp|mcheck|alloca|link|elf)[^>"]*)[>"]')
lin = 0; quoted=0
for r in rows:
    d = r.get("real_deps") or ""
    hs = re.findall(r'#include\s*([<"][^>"]+[>"])', d)
    for h in set(hs): inc[h]+=1
    if linuxish.search(d): lin += 1
    if re.search(r'#include\s*"', d): quoted += 1
print("top includes:", inc.most_common(40))
print("rows w/ linux/glibc-ish include:", lin, f"{lin/n:.1%}")
print("rows w/ quoted include:", quoted)
static = sum(1 for r in rows if r["func_def"].lstrip().startswith("static") or " static " in r["func_def"].split("(")[0])
print("static funcs:", static, f"{static/n:.1%}")
L = sorted(len(r["func_def"].splitlines()) for r in rows)
print("func_def lines median/p90/max:", L[n//2], L[int(n*.9)], L[-1])
types = collections.Counter()
for r in rows:
    spec = r["real_iospec"]
    spec = json.loads(spec) if isinstance(spec,str) else spec
    for v in (spec.get("typemap") or {}).values(): types[v.split("[")[0] if isinstance(v,str) else str(v)]+=1
print("typemap types:", types.most_common(20))
print("error fields non-empty:", sum(1 for r in rows if r.get("real_error")), sum(1 for r in rows if r.get("real_io_error")))
