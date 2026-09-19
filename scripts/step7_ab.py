#!/usr/bin/env python3
"""step7: reporter-shaped A -> B -> B(with fresh cache_salt), token-id prompts, natural content.
Prints one line per request in execution order (matches FBW request order)."""
import argparse, json, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--grid", type=int, default=816)
ap.add_argument("--N", type=int, default=20); ap.add_argument("--r", type=int, required=True)
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--a-max", type=int, default=300)
ap.add_argument("--b-max", type=int, default=64); ap.add_argument("--suffix", type=int, default=2000)
ap.add_argument("--label", default="")
a = ap.parse_args()
PH = [[2374,5810,1122,8391],[740,2029,5810,1122],[3312,740,2029,91],[8391,3312,740,2029],
      [1122,8391,3312,5810],[91,2374,5810,740]]
def natural(n, s):
    x = (s*2654435761+1) & 0xFFFFFFFF; o = []
    while len(o) < n:
        x = (1103515245*x+12345) & 0x7FFFFFFF; o.extend(PH[x % len(PH)])
        if (x >> 8) % 3 == 0 and len(o) >= 8: o.extend(o[-8:])
    return o[:n]
def metric(n):
    t = 0.0
    for l in urllib.request.urlopen(a.base+"/metrics", timeout=30).read().decode().splitlines():
        if l.startswith(n) and not l.startswith("#"):
            try: t += float(l.rsplit(" ", 1)[-1])
            except ValueError: pass
    return t
def run(tag, ids, mx, salt=None):
    p = {"model":"qwen38","prompt":ids,"max_tokens":mx,"temperature":0,"logprobs":1,
         "return_tokens_as_token_ids":True}
    if salt: p["cache_salt"] = salt
    c0 = metric("vllm:corrupted_requests"); d0 = metric("vllm:spec_decode_num_drafts")
    h0 = metric("vllm:prefix_cache_hits")
    try:
        r = urllib.request.urlopen(urllib.request.Request(a.base+"/v1/completions",
            data=json.dumps(p).encode(), headers={"Content-Type":"application/json"}), timeout=600)
        ch = json.loads(r.read())["choices"][0]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:120]; time.sleep(0.5)
        cd = metric("vllm:corrupted_requests")-c0
        print(f"{a.label} {tag} len={len(ids)} len%grid={len(ids)%a.grid} HTTP{e.code} corrupted+{cd:.0f} "
              f"prefix_hit_tokens+{metric('vllm:prefix_cache_hits')-h0:.0f} {body}", flush=True); return []
    lp = ch.get("logprobs") or {}
    out = [int(t.split(":",1)[1]) for t in (lp.get("tokens") or []) if t.startswith("token_id:")]
    tl = lp.get("token_logprobs") or []
    nonfinite = any((x is None) or (x != x) or x == float("-inf") for x in tl)
    cd = metric("vllm:corrupted_requests")-c0; dd = metric("vllm:spec_decode_num_drafts")-d0
    hits = metric("vllm:prefix_cache_hits")-h0
    txt = ch.get("text","")[:16]
    print(f"{a.label} {tag} len={len(ids)} len%grid={len(ids)%a.grid} out={len(out)} corrupted+{cd:.0f} "
          f"nonfinite_lp={nonfinite} drafts+{dd:.0f} prefix_hit_tokens+{hits:.0f} text={txt!r}", flush=True)
    return out
for s in range(a.seeds):
    plen = a.N*a.grid + a.r
    A = natural(plen, s*9973 + a.N*101 + a.r)
    ao = run(f"r={a.r} N={a.N} seed={s} A", A, a.a_max)
    B = A + ao + natural(a.suffix, s*131+1)
    run(f"r={a.r} N={a.N} seed={s} B", B, a.b_max)
    run(f"r={a.r} N={a.N} seed={s} Bsalt", B, a.b_max, salt=f"salt-{a.r}-{s}-{time.time_ns()}")
print("STEP7_DONE", flush=True)
