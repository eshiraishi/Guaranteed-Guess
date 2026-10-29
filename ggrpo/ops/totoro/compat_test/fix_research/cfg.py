import json,sys,shutil,os,tempfile,warnings
import transformers
from transformers import AutoConfig
src=os.path.expanduser("~/ggrpo/runs/probe_full")
variants={}
c=json.load(open(src+"/config.json"))
variants["raw"]=dict(c)
v=dict(c); v["rope_theta"]=1e6; v["torch_dtype"]="float32"; variants["dual"]=v
v=dict(c); v["rope_theta"]=1e6; v["torch_dtype"]="float32"; v["sliding_window"]=32768; v["bos_token_id"]=151643; v.pop("layer_types"); variants["dual_full"]=v
for k,cfg in variants.items():
    d=tempfile.mkdtemp(dir=os.path.expanduser("~/compat_test/fix_research")); json.dump(cfg,open(d+"/config.json","w"))
    ac=AutoConfig.from_pretrained(d)
    rp=getattr(ac,"rope_parameters",None)
    print(transformers.__version__,k,"rope_theta attr=",getattr(ac,"rope_theta",None),"rope_parameters=",rp,"sw=",ac.sliding_window,"dtype=",getattr(ac,"torch_dtype",None) if transformers.__version__.startswith("4") else ac.dtype, "attn_impl=",ac._attn_implementation)
    # rotary inv_freq
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    r=Qwen2RotaryEmbedding(ac); print("   inv_freq[1]=",float(r.inv_freq[1]), "expected(1e6)=",1e6**(-2/128), "(1e4)=",1e4**(-2/128))
