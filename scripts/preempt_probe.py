#!/usr/bin/env python3
"""Force real preemption + resume and look for a K+1-token recompute tail
(work item C.2).

Shape: `--nfillers` requests are submitted first and hold KV while they decode;
the victim is submitted LAST, so under the FCFS policy it is `self.running[-1]`
and therefore the request the scheduler preempts when blocks run out
(scheduler.py:783-784). The victim has already produced output tokens at that
point, so on resume `num_output_tokens > 0` and #47123's context-phase test
does not classify it as a context request.

The victim's recompute ends at `num_prompt_tokens + num_output_tokens`, so the
last recompute chunk is `(P + G) mod block_size` (see REVIEW_FOLLOWUP.md for
the file:line argument). `--victim-offset` shifts P so that this residue can be
steered onto K+1 once G has been measured in a calibration run.

Everything the guard sees is logged on the server side by the FBW META line
(resumed=/nout=/ctx=/nctx=/fud=), not here.
"""
import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--grid", type=int, default=816)
ap.add_argument("--nfillers", type=int, default=4)
ap.add_argument("--filler-blocks", type=int, default=20)
ap.add_argument("--filler-max", type=int, default=400)
ap.add_argument("--victim-blocks", type=int, default=20)
ap.add_argument("--victim-offset", type=int, default=0,
                help="victim prompt = victim_blocks*grid + offset")
ap.add_argument("--victim-max", type=int, default=600)
ap.add_argument("--delay", type=float, default=6.0,
                help="seconds between the fillers and the victim")
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

PH = [[2374, 5810, 1122, 8391], [740, 2029, 5810, 1122], [3312, 740, 2029, 91],
      [8391, 3312, 740, 2029], [1122, 8391, 3312, 5810], [91, 2374, 5810, 740]]


def natural(n, s):
    x = (s * 2654435761 + 1) & 0xFFFFFFFF
    o = []
    while len(o) < n:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        o.extend(PH[x % len(PH)])
        if (x >> 8) % 3 == 0 and len(o) >= 8:
            o.extend(o[-8:])
    return o[:n]


def h(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def metric(name):
    t = 0.0
    body = urllib.request.urlopen(a.base + "/metrics", timeout=30).read().decode()
    for line in body.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                t += float(line.rsplit(" ", 1)[-1])
            except ValueError:
                pass
    return t


lock = threading.Lock()
out_fh = open(a.out, "a")
results = []


def fire(role, ids, mx):
    rec = {"label": a.label, "role": role, "prompt_tokens": len(ids),
           "prompt_len_mod_grid": len(ids) % a.grid, "input_sha256": h(ids),
           "max_tokens": mx, "t_start": round(time.time(), 3)}
    payload = {"model": "qwen38", "prompt": ids, "max_tokens": mx,
               "temperature": 0, "logprobs": 1, "return_tokens_as_token_ids": True}
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(
                a.base + "/v1/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}), timeout=1800)
        ch = json.loads(resp.read())["choices"][0]
        rec["http_status"] = 200
        lp = ch.get("logprobs") or {}
        toks = [int(t.split(":", 1)[1]) for t in (lp.get("tokens") or [])
                if t.startswith("token_id:")]
        tl = lp.get("token_logprobs") or []
        rec["output_tokens"] = len(toks)
        rec["output_sha256"] = h(toks) if toks else ""
        rec["nonfinite_logprob"] = any(
            (x is None) or (x != x) or x == float("-inf") for x in tl)
        rec["text_head"] = ch.get("text", "")[:24]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:160]
        rec.update(http_status=e.code, http_body=body, output_tokens=0,
                   output_sha256="", nonfinite_logprob="nan" in body.lower(),
                   text_head="")
    except Exception as e:
        rec.update(http_status=-1, http_body=repr(e)[:160], output_tokens=0,
                   output_sha256="", nonfinite_logprob=False, text_head="")
    rec["t_end"] = round(time.time(), 3)
    rec["corrupted"] = bool(
        (rec["http_status"] == 400 and "nan" in rec.get("http_body", "").lower())
        or rec["nonfinite_logprob"] is True)
    with lock:
        out_fh.write(json.dumps(rec) + "\n")
        out_fh.flush()
        results.append(rec)
        print("%s %s len=%d mod=%d out=%d http=%s corrupted=%s"
              % (a.label, role, len(ids), len(ids) % a.grid, rec["output_tokens"],
                 rec["http_status"], rec["corrupted"]), flush=True)


c0 = metric("vllm:corrupted_requests")
p0 = metric("vllm:num_preemptions")
print("%s START preemptions=%.0f corrupted=%.0f" % (a.label, p0, c0), flush=True)

threads = []
for i in range(a.nfillers):
    ids = natural(a.filler_blocks * a.grid, 4000 + i)
    t = threading.Thread(target=fire, args=("filler%d" % i, ids, a.filler_max))
    t.start()
    threads.append(t)
    time.sleep(0.4)

time.sleep(a.delay)
vids = natural(a.victim_blocks * a.grid + a.victim_offset, 9100)
tv = threading.Thread(target=fire, args=("victim", vids, a.victim_max))
tv.start()
threads.append(tv)

for t in threads:
    t.join()

p1 = metric("vllm:num_preemptions")
c1 = metric("vllm:corrupted_requests")
summary = {"label": a.label, "role": "SUMMARY",
           "preemptions_total_before": p0, "preemptions_total_after": p1,
           "preemptions_delta": p1 - p0,
           "corrupted_total_before": c0, "corrupted_total_after": c1,
           "corrupted_delta": c1 - c0,
           "victim_prompt_tokens": len(vids),
           "victim_offset": a.victim_offset}
out_fh.write(json.dumps(summary) + "\n")
out_fh.flush()
print("%s DONE preemptions+%.0f corrupted+%.0f victim_plen=%d"
      % (a.label, p1 - p0, c1 - c0, len(vids)), flush=True)
