import duckdb, time, collections
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs; SET http_retries=10; SET http_retry_wait_ms=20000; SET http_retry_backoff=2;")
agg = collections.Counter(); probs = collections.defaultdict(set)
for i in range(36):
    url = 'https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/main/data/train-%05d-of-00036.parquet' % i
    for attempt in range(6):
        try:
            rows = con.execute(f"SELECT programmingLanguage, verdict, problem_id FROM read_parquet('{url}') WHERE programmingLanguage IN ('GNU C11','GNU C')").fetchall()
            break
        except Exception as e:
            print('retry', i, str(e)[:100], flush=True); time.sleep(90)
    for lang, v, p in rows:
        agg[(lang, v)] += 1; probs[(lang, v)].add(p)
    print('done', i, sum(agg.values()), flush=True)
    time.sleep(10)
for k, n in agg.most_common(): print(k, n, len(probs[k]))
allok = set().union(*[s for (l, v), s in probs.items() if v == 'OK']) if probs else set()
print('distinct problems with OK C submission', len(allok))
