import gzip, json, sys, statistics, collections
rows = [json.loads(l) for l in gzip.open(sys.argv[1], 'rt')]
print('tasks', len(rows), 'keys', list(rows[0].keys()))
b = [len(r['base_input']) for r in rows]; p = [len(r['plus_input']) for r in rows]
print('base inputs mean/median', statistics.mean(b), statistics.median(b))
print('plus inputs mean/median/min/max', round(statistics.mean(p),1), statistics.median(p), min(p), max(p))
def kind(x):
    if isinstance(x, bool): return 'bool'
    if isinstance(x, int): return 'int'
    if isinstance(x, float): return 'float'
    if isinstance(x, str): return 'str'
    if isinstance(x, list):
        inner = {kind(e) for e in x}
        return 'list[' + ','.join(sorted(inner)) + ']' if len(inner) <= 1 else 'list[mixed]'
    return type(x).__name__
c = collections.Counter()
simple = 0
for r in rows:
    ks = {kind(a) for inp in r['base_input'][:1] for a in inp}
    c[tuple(sorted(ks))] += 1
    if ks <= {'int', 'float', 'bool', 'str', 'list[int]', 'list[float]', 'list[str]', 'list[]'}: simple += 1
print('tasks whose args are scalars/str/flat lists', simple)
print(c.most_common(12))
ids = sorted(int(r['task_id'].split('/')[1]) for r in rows)
print('task id range', ids[0], ids[-1])
