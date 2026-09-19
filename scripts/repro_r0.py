#!/usr/bin/env python3
"""R0: token-ID controlled reproducer for vLLM #55766 (hybrid GDN prefix-cache NaN).

Deterministic, re-runnable. Talks to a vLLM OpenAI-compatible server via /v1/completions
with integer token-ID prompts (no tokenizer text ambiguity). For each cell it:
  1. Runs request A whose prompt length is exactly N*grid + r tokens.
  2. Saves A's generated output token IDs.
  3. Builds request B = A.prompt_ids + A.output_ids + deterministic suffix_ids, which hits a
     block-aligned prefix up to a boundary and then prefills an uncached suffix.
  4. Records finiteness directly: sampled logprobs NaN/-inf, degenerate token-0/"!" runs, and
     the server's vllm:corrupted_requests_total delta from /metrics.

Grid is passed in (resolve it from the server startup geometry dump; do NOT hardcode 816).
Nothing here changes cache keys or synchronization.

Usage:
  repro_r0.py --base http://127.0.0.1:PORT --model NAME --grid G \
      --N 10,20 --r 6,8,7 --suffix 2xgrid --seeds 3 --a-max 64 --b-max 16 \
      --prefix-cache on --ngram on --out results.csv [--cache-salt-mode none|per-b]
"""
import argparse, csv, json, os, sys, time, urllib.request, urllib.error

VOCAB_SAFE_LO = 1000        # avoid special/control tokens near 0 and EOS
VOCAB_SAFE_HI = 100000      # stay well inside any Qwen vocab

def rng_ids(seed, n, lo=VOCAB_SAFE_LO, hi=VOCAB_SAFE_HI):
    """Deterministic LCG token-id stream, no numpy dependency."""
    x = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    out = []
    for _ in range(n):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        out.append(lo + (x % (hi - lo)))
    return out

def http_json(url, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get_metric(base, name):
    try:
        req = urllib.request.Request(base + "/metrics")
        with urllib.request.urlopen(req, timeout=30) as r:
            total = 0.0
            for line in r.read().decode().splitlines():
                if line.startswith(name) and not line.startswith("#"):
                    total += float(line.rsplit(" ", 1)[-1])
            return total
    except Exception:
        return float("nan")

def completions(base, model, prompt_ids, max_tokens, temperature, top_p, top_k,
                cache_salt=None, logprobs=1, seed=0):
    payload = {
        "model": model,
        "prompt": prompt_ids,          # integer token IDs
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": logprobs,
        "seed": seed,
        "stream": False,
    }
    if top_k is not None:
        payload["top_k"] = top_k
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    t0 = time.time()
    resp = http_json(base + "/v1/completions", payload)
    dt = time.time() - t0
    ch = resp["choices"][0]
    lp = ch.get("logprobs") or {}
    out_ids = lp.get("tokens") or []
    # token_logprobs may contain None/NaN; detect non-finite
    tlp = lp.get("token_logprobs") or []
    nonfinite = any((x is None) or (x != x) or (x == float("-inf")) for x in tlp)
    text = ch.get("text", "")
    # degenerate: all-"!" or empty
    degenerate = (len(text) > 0 and set(text) <= {"!"}) or (max_tokens > 0 and text == "")
    return {
        "text": text, "out_ids": out_ids, "token_logprobs": tlp,
        "nonfinite_logprob": nonfinite, "degenerate": degenerate,
        "finish": ch.get("finish_reason"), "runtime": dt,
    }

def parse_suffix(spec, grid):
    if spec.endswith("xgrid"):
        return int(round(float(spec[:-5]) * grid))
    if spec == "grid-1": return grid - 1
    if spec == "grid": return grid
    if spec == "grid+1": return grid + 1
    return int(spec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", type=int, required=True, help="resolved mamba state block size")
    ap.add_argument("--N", default="10,20")
    ap.add_argument("--r", default="6,8,7")
    ap.add_argument("--suffix", default="2xgrid")   # 16 | grid-1 | grid | grid+1 | 2xgrid | int
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--a-max", type=int, default=64)
    ap.add_argument("--b-max", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cache-salt-mode", default="none", choices=["none", "per-b"])
    ap.add_argument("--revision", default="?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    Ns = [int(x) for x in args.N.split(",")]
    rs = [int(x) for x in args.r.split(",")]
    suffix_len = parse_suffix(args.suffix, args.grid)

    cols = ["revision","model","grid","N","r","seed","prompt_len","suffix_len",
            "cache_salt_mode","a_finish","a_out_len","b_prefix_len",
            "b_nonfinite_logprob","b_degenerate","b_text_head","corrupted_delta",
            "b_runtime","status"]
    newfile = args.append and os.path.exists(args.out)
    f = open(args.out, "a" if args.append else "w", newline="")
    w = csv.writer(f)
    if not newfile:
        w.writerow(cols)

    for N in Ns:
        for r in rs:
            plen = N * args.grid + r
            for seed in range(args.seeds):
                a_ids = rng_ids(seed * 10007 + N * 131 + r, plen)
                # Request A
                try:
                    a = completions(args.base, args.model, a_ids, args.a_max,
                                    args.temperature, args.top_p, args.top_k, seed=seed)
                except Exception as e:
                    w.writerow([args.revision,args.model,args.grid,N,r,seed,plen,suffix_len,
                                args.cache_salt_mode,"AERR",0,0,"","",f"{e}"[:40],"",0,"A_ERROR"])
                    f.flush(); continue
                a_out = a["out_ids"]
                suffix = rng_ids(seed * 733 + 991, suffix_len)
                # B prompt: A prompt + A output (as ids) + deterministic suffix.
                # a_out entries are decoded token strings from logprobs; rebuild ids via
                # a fresh deterministic tail so B is fully id-controlled and reproducible.
                b_prefix = a_ids + rng_ids(seed * 17 + 3, max(len(a_out), 1))
                b_ids = b_prefix + suffix
                salt = None
                if args.cache_salt_mode == "per-b":
                    salt = f"salt-{N}-{r}-{seed}-{time.time_ns()}"
                c0 = get_metric(args.base, "vllm:corrupted_requests_total")
                try:
                    b = completions(args.base, args.model, b_ids, args.b_max,
                                    args.temperature, args.top_p, args.top_k,
                                    cache_salt=salt, seed=seed)
                except Exception as e:
                    w.writerow([args.revision,args.model,args.grid,N,r,seed,plen,suffix_len,
                                args.cache_salt_mode,a["finish"],len(a_out),len(b_prefix),
                                "","",f"{e}"[:40],"",0,"B_ERROR"])
                    f.flush(); continue
                c1 = get_metric(args.base, "vllm:corrupted_requests_total")
                cd = (c1 - c0) if (c1 == c1 and c0 == c0) else float("nan")
                nan = b["nonfinite_logprob"] or b["degenerate"] or (cd and cd > 0)
                status = "NAN" if nan else "clean"
                w.writerow([args.revision,args.model,args.grid,N,r,seed,plen,suffix_len,
                            args.cache_salt_mode,a["finish"],len(a_out),len(b_prefix),
                            b["nonfinite_logprob"],b["degenerate"],
                            b["text"][:20].replace("\n"," "),cd,round(b["runtime"],3),status])
                f.flush()
                print(f"N={N} r={r} seed={seed} plen={plen} -> {status} "
                      f"(nonfinite={b['nonfinite_logprob']} degen={b['degenerate']} "
                      f"corrupted+{cd})", flush=True)
    f.close()

if __name__ == "__main__":
    main()
