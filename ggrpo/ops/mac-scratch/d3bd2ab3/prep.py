import pandas as pd, os, sys
from datasets import load_from_disk
S=sys.argv[1]
df=pd.read_parquet('rl/artifacts/data/humaneval_armv8.parquet')
off=df[df.x86_origin=='gg-official']
ds=load_from_disk('data_cache/decompile-eval/humaneval')
rows=[r for r in ds if r['language']=='c' and r['opt']=='O0']; rows.sort(key=lambda r:int(r['index']))
os.makedirs(S+'/cmp/src',exist_ok=True); os.makedirs(S+'/cmp/off',exist_ok=True)
n=0
for _,r in off.iterrows():
    k=int(r.task_id.split('/')[-1]); row=rows[k]
    name=f"{k}_{r.opt}"
    open(f"{S}/cmp/src/{name}.c",'w').write(row['func_dep']+"\n"+row['func'])
    open(f"{S}/cmp/off/{name}.x86.s",'w').write(r.x86); open(f"{S}/cmp/off/{name}.arm.s",'w').write(r.arm_ref); n+=1
print(n, df[df.x86_origin=='gg-official'].opt.value_counts().to_dict())
