import duckdb
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
q = """
SELECT programmingLanguage, verdict, count(*) n, count(DISTINCT problem_id) np
FROM read_parquet([%s])
WHERE programmingLanguage IN ('GNU C11','GNU C')
GROUP BY 1,2 ORDER BY n DESC
""" % ",".join("'https://huggingface.co/datasets/open-r1/codeforces-submissions/resolve/main/data/train-%05d-of-00036.parquet'" % i for i in range(36))
for r in con.execute(q).fetchall(): print(r)
