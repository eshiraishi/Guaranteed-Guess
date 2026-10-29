import json, sys, collections
angha = set(open(sys.argv[2]).read().split())
rows = [json.loads(l)["text"] for l in open(sys.argv[1])]
repos = collections.Counter()
hit = 0
users=set(); rps=set()
for r in rows:
    parts = r["path"].split("/")
    user, repo = parts[0], parts[1]
    users.add(user); rps.add((user,repo))
    if repo in angha: hit += 1; repos[repo]+=1
print("rows", len(rows), "distinct repos", len(rps), "distinct users", len(users))
print("rows whose repo name is in public AnghaBench top-level:", hit, repos.most_common(20))
print("sample paths:", [r["path"] for r in rows[:10]])
