import pandas as pd, re, collections, glob
D='rl/artifacts/data/'
df=pd.read_parquet(D+'humaneval_armv8.parquet')
print(list(df.columns)); print(df['x86_origin'].value_counts())
def grab(s,pat):
    return re.findall(pat,s or '',re.M)
for g,sub in [('gg-official',df[df.x86_origin=='gg-official']),('other',df[df.x86_origin!='gg-official'])]:
    for col in ['x86','arm_ref']:
        if col not in df.columns: print('no col',col); continue
        for nm,pat in [('build_version',r'^\s*\.build_version.*$'),('ident',r'^\s*\.ident.*$')]:
            c=collections.Counter(x for s in sub[col] for x in set(grab(s,pat)))
            print(g,col,nm,len(sub),c.most_common(4))
for f in sorted(glob.glob(D+'*.parquet')):
    d=pd.read_parquet(f); print(f,len(d),[c for c in d.columns if 'origin' in c or 'opt' in c])
    if 'x86_origin' in d: print(d['x86_origin'].value_counts().to_dict())
