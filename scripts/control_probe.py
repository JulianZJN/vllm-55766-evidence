#!/usr/bin/env python3
import argparse, json, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", default="qwen38")
ap.add_argument("--plen", type=int, default=16326); ap.add_argument("--max", type=int, default=200)
ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--content", default="natural")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--label", default="")
a = ap.parse_args()
def metric(n):
    t = 0.0
    for l in urllib.request.urlopen(a.base + "/metrics", timeout=30).read().decode().splitlines():
        if l.startswith(n) and not l.startswith("#"):
            try: t += float(l.rsplit(" ", 1)[-1])
            except: pass
    return t
def natural(n, s=1):
    ph = [[2374,5810,1122,8391],[740,2029,5810,1122],[3312,740,2029,91],[8391,3312,740,2029]]
    x=(s*2654435761+1)&0xFFFFFFFF; o=[]
    while len(o)<n:
        x=(1103515245*x+12345)&0x7FFFFFFF; o.extend(ph[x%len(ph)])
        if (x>>8)%3==0 and len(o)>=8: o.extend(o[-8:])
    return o[:n]
def ultrarep(n,s=1):
    ph=[[2374,5810,1122,8391],[740,2029,5810,1122],[3312,740,2029,91]][s%3]
    o=(ph*((n//4)+1))[:n]; return o
def rand(n,s=1):
    x=(s*40503+7)&0xFFFFFFFF; o=[]
    for _ in range(n):
        x=(1103515245*x+12345)&0x7FFFFFFF; o.append(1000+x%99000)
    return o
ids = {"natural":natural,"random":rand,"ultrarep":ultrarep}[a.content](a.plen, a.seed)
c0=metric("vllm:corrupted_requests"); d0=metric("vllm:spec_decode_num_drafts"); ac0=metric("vllm:spec_decode_num_accepted_tokens")
p={"model":a.model,"prompt":ids,"max_tokens":a.max,"temperature":a.temp}
if a.temp>0: p.update({"top_p":0.95,"top_k":20})
r=urllib.request.urlopen(urllib.request.Request(a.base+"/v1/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"}),timeout=180)
txt=json.loads(r.read())["choices"][0]["text"][:24]
cd=metric("vllm:corrupted_requests")-c0; dd=metric("vllm:spec_decode_num_drafts")-d0; ad=metric("vllm:spec_decode_num_accepted_tokens")-ac0
status="NAN" if cd>0 else ("degen" if (txt and set(txt)<=set("! ")) else "clean")
print(f"{a.label} seed={a.seed} plen={a.plen} content={a.content} temp={a.temp} -> {status} corrupted+{cd:.0f} drafts+{dd:.0f} accepts+{ad:.0f} out={txt!r}")
