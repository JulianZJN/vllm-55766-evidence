#!/usr/bin/env python3
"""Frozen-input paired A / B / B-salted control (work item B).

Difference from step7_ab.py: B is NOT built from the reply the server under
test produced for A. Prompt ids, the "reply" continuation and the suffix are
all generated deterministically from the seed, so the *identical* token-id
lists are sent to every arm (unfixed / fixed). Nothing in the input depends on
the build, so the two arms are a paired comparison on frozen inputs.

A's prompt ids use the same generator and the same seeding as step7_ab.py, so
A is bit-identical to the published step-7 runs. Only B's middle section
changes (frozen synthetic continuation instead of A's own reply).

Writes one JSON object per request to --out (append) and one line per request
to stdout. Every request's token-id list is hashed (sha256 of the decimal ids
joined by commas) so anyone can tell which runs shared inputs.
"""
import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--grid", type=int, default=816)
ap.add_argument("--N", type=int, default=20)
ap.add_argument("--r", type=int, required=True)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--a-max", type=int, default=300)
ap.add_argument("--b-max", type=int, default=64)
ap.add_argument("--reply", type=int, default=300, help="frozen continuation len")
ap.add_argument("--suffix", type=int, default=2000)
ap.add_argument("--build", required=True, help="unfixed | fixed")
ap.add_argument("--cell", required=True, help="cell id, e.g. B-frozen-r6-N20")
ap.add_argument("--out", required=True)
a = ap.parse_args()

# identical to step7_ab.py
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


out_fh = open(a.out, "a")


def run(role, seed, ids, mx, salt=None):
    rec = {
        "cell": a.cell, "build": a.build, "N": a.N, "r": a.r, "seed": seed,
        "role": role, "prompt_tokens": len(ids),
        "prompt_len_mod_grid": len(ids) % a.grid,
        "input_sha256": h(ids), "cache_salt": salt or "",
        "max_tokens": mx,
    }
    payload = {"model": "qwen38", "prompt": ids, "max_tokens": mx,
               "temperature": 0, "logprobs": 1, "return_tokens_as_token_ids": True}
    if salt:
        payload["cache_salt"] = salt
    c0 = metric("vllm:corrupted_requests")
    hit0 = metric("vllm:prefix_cache_hits")
    d0 = metric("vllm:spec_decode_num_drafts")
    p0 = metric("vllm:num_preemptions")
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(
                a.base + "/v1/completions", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}), timeout=900)
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
        rec["http_status"] = e.code
        rec["http_body"] = body
        rec["output_tokens"] = 0
        rec["output_sha256"] = ""
        rec["nonfinite_logprob"] = "nan" in body.lower()
        rec["text_head"] = ""
        time.sleep(0.5)
    rec["corrupted_delta"] = metric("vllm:corrupted_requests") - c0
    rec["prefix_hit_tokens_delta"] = metric("vllm:prefix_cache_hits") - hit0
    rec["drafts_delta"] = metric("vllm:spec_decode_num_drafts") - d0
    rec["preemptions_delta"] = metric("vllm:num_preemptions") - p0
    # "corrupted" = the engine's own NaN counter moved, or the request came
    # back as HTTP 400 because a NaN logprob is not JSON-encodable.
    rec["corrupted"] = bool(rec["corrupted_delta"] > 0
                            or (rec["http_status"] == 400
                                and "nan" in rec.get("http_body", "").lower())
                            or rec["nonfinite_logprob"] is True)
    out_fh.write(json.dumps(rec) + "\n")
    out_fh.flush()
    print("%s %s seed=%d %s len=%d mod=%d corrupted=%s http=%s hit=%d sha=%s"
          % (a.cell, a.build, seed, role, len(ids), len(ids) % a.grid,
             rec["corrupted"], rec["http_status"],
             rec["prefix_hit_tokens_delta"], rec["input_sha256"][:12]),
          flush=True)


for s in range(a.seeds):
    plen = a.N * a.grid + a.r
    A = natural(plen, s * 9973 + a.N * 101 + a.r)      # same ids as step7
    REPLY = natural(a.reply, 777000 + s)               # frozen, build-independent
    SUF = natural(a.suffix, s * 131 + 1)               # same ids as step7
    B = A + REPLY + SUF
    run("A", s, A, a.a_max)
    run("B", s, B, a.b_max)
    run("B-salted", s, B, a.b_max, salt="frozen-%d-%d-%d" % (a.r, s, a.N))
print("FROZEN_DONE", flush=True)
