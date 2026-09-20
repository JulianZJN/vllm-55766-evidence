#!/usr/bin/env python3
"""Build the per-request result tables in ../results/ from the raw logs.

Inputs
  ../traces/step7/wl_<build>_<cell>.txt   one line per request (step7_ab.py)
  ../traces/step7/fbw_<build>_<cell>.log.gz   FBW META/P1/P2/P3 lines
  ../results/frozen_control_requests.jsonl    one JSON per request (frozen_ab.py)
  ../traces/frozen/fbw_frozen_<build>.log.gz
Outputs
  ../results/step7_requests.csv
  ../results/frozen_control_requests.csv
  ../results/input_hashes.csv

Every row keeps its own (cell, build, N, r, seed, role) denominator; nothing is
aggregated here.

Token-id inputs are regenerated with the same deterministic generator the
drivers use, so the sha256 of each request's input token-id list can be
recomputed by anyone from this file alone. For the step-7 B / B-salted rows the
hash is NOT reproducible: B was built from the reply that *that* arm's server
produced for A, so its ids differ between arms (this is exactly what the frozen
control in frozen_control_requests.csv removes).
"""
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
TR = os.path.join(PACK, "traces")
RES = os.path.join(PACK, "results")

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


def sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


# ---------------------------------------------------------------- FBW parsing
def fbw_lines(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as fh:
        for line in fh:
            i = line.find("FBW step=")
            if i >= 0:
                yield line[i:].rstrip("\n")


def parse_fbw(path):
    """-> list of per-request dicts in execution order.

    Each dict: req (last 6 chars of the id), prefill_chunks [(ntok, mode)],
    tail_len, tail_mode, tail_step, resumed_steps [...].
    """
    steps = {}
    for line in fbw_lines(path):
        m = re.match(r"FBW step=(\d+) META ", line)
        if not m:
            continue
        s = int(m.group(1))
        d = {"mode": re.search(r"mode=(\w+)", line).group(1),
             "uniform": re.search(r"uniform=(\w+)", line).group(1),
             "reqs": re.findall(r"'([0-9a-f]{6})'", line[:line.find("owned=")]),
             "sched": ast.literal_eval(re.search(r"sched=(\[[^\]]*\])", line).group(1)),
             "k": ast.literal_eval(re.search(r" k=(\[[^\]]*\])", line).group(1)),
             "npf": int(re.search(r"npf=(\d+)", line).group(1))}
        for f in ("resumed", "nout", "ctx", "nctx", "fud", "ncomp"):
            mm = re.search(r"\b%s=(\[[^\]]*\]|[^ ]+)" % f, line)
            if mm:
                d[f] = mm.group(1)
        if len(d["reqs"]) > 1:
            raise ValueError(f"{path}: step {s} has multiple requests; serial parser cannot attribute rows")
        if len(d["sched"]) != len(d["reqs"]) or len(d["k"]) != len(d["reqs"]):
            raise ValueError(f"{path}: inconsistent per-request metadata at step {s}")
        if s in steps and steps[s] != d:
            raise ValueError(f"{path}: conflicting META records for step {s}")
        steps[s] = d
    order, cur = [], None
    for s in sorted(steps):
        d = steps[s]
        if not d["reqs"]:
            continue
        rid = d["reqs"][0]
        if cur is None or cur["req"] != rid:
            cur = {"req": rid, "prefill_chunks": [], "steps": []}
            order.append(cur)
        cur["steps"].append(s)
        if d["npf"] > 0:
            cur["prefill_chunks"].append((d["sched"][0], d["mode"], s))
    for r in order:
        if r["prefill_chunks"]:
            n, mode, s = r["prefill_chunks"][-1]
            r["tail_len"], r["tail_mode"], r["tail_step"] = n, mode, s
        else:
            r["tail_len"] = r["tail_mode"] = r["tail_step"] = ""
    return order


# ------------------------------------------------------------------- step 7
WL = re.compile(
    r"^(?P<build>\w+) r=(?P<r>\d+) N=(?P<N>\d+) seed=(?P<seed>\d+) (?P<role>\S+) "
    r"len=(?P<len>\d+)(?P<rest>.*)$")


def step7_rows():
    d = os.path.join(TR, "step7")
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("wl_") or not fn.endswith(".txt"):
            continue
        cell = fn[3:-4]                      # e.g. unfixed_r6n34
        fbw = os.path.join(d, "fbw_%s.log.gz" % cell)
        per_req = parse_fbw(fbw)
        with open(os.path.join(d, fn)) as driver:
            request_count = sum(bool(WL.match(line.strip())) for line in driver)
        _require_serial_trace_alignment(per_req, request_count, cell)
        i = 0
        for line in open(os.path.join(d, fn)):
            m = WL.match(line.strip())
            if not m:
                continue
            g = m.groupdict()
            rest = g["rest"]
            http, corrupted, outcome = _parse_step7_outcome(rest)
            corr = re.search(r"corrupted\+(\d+)", rest)
            hit = re.search(r"prefix_hit_tokens\+(\d+)", rest)
            fb = per_req[i] if i < len(per_req) else {}
            i += 1
            seed = int(g["seed"])
            N, r = int(g["N"]), int(g["r"])
            if g["role"] == "A":
                ids = natural(N * 816 + r, seed * 9973 + N * 101 + r)
                ihash, note = sha(ids), ""
            else:
                ihash = ""
                note = "not frozen: B = A ids + THIS arm's reply + suffix"
            rows.append({
                "cell": "step7-%s" % cell, "build": g["build"], "N": N, "r": r,
                "seed": seed, "role": g["role"],
                "prompt_tokens": int(g["len"]),
                "prompt_len_mod_grid": int(g["len"]) % 816,
                "tail_chunk_len": fb.get("tail_len", ""),
                "tail_step_graph_mode": fb.get("tail_mode", ""),
                "prefill_chunks": "|".join(
                    "%d:%s" % (n, mo) for n, mo, _ in fb.get("prefill_chunks", [])),
                "corrupted": int(corrupted),
                "request_outcome": outcome,
                "http_status": http,
                "corrupted_counter_delta": corr.group(1) if corr else "",
                "prefix_hit_tokens": hit.group(1) if hit else "",
                "input_sha256": ihash, "input_note": note,
            })
    return rows


# ------------------------------------------------------------- frozen control
def frozen_rows(jsonl="frozen_control_requests.jsonl", trace_dir="frozen",
                note="frozen: identical ids sent to every arm"):
    p = os.path.join(RES, jsonl)
    if not os.path.exists(p):
        return []
    fbw = {}
    fd = os.path.join(TR, trace_dir)
    if os.path.isdir(fd):
        for fn in sorted(os.listdir(fd)):
            if fn.endswith(".log.gz"):
                build = fn.replace("fbw_frozen_", "").replace(".log.gz", "")
                fbw[build] = parse_fbw(os.path.join(fd, fn))
    seen = {}
    rows = []
    for line in open(p):
        rec = json.loads(line)
        b = rec["build"]
        i = seen.get(b, 0)
        seen[b] = i + 1
        fb = fbw.get(b, [])
        if i >= len(fb):
            raise ValueError(f"{jsonl}: no trace request {i} for build {b!r}; check the trace filename mapping")
        f = fb[i]
        rows.append({
            "cell": rec["cell"], "build": b, "N": rec["N"], "r": rec["r"],
            "seed": rec["seed"], "role": rec["role"],
            "prompt_tokens": rec["prompt_tokens"],
            "prompt_len_mod_grid": rec["prompt_len_mod_grid"],
            "tail_chunk_len": f.get("tail_len", ""),
            "tail_step_graph_mode": f.get("tail_mode", ""),
            "prefill_chunks": "|".join(
                "%d:%s" % (n, mo) for n, mo, _ in f.get("prefill_chunks", [])),
            "corrupted": int(rec["corrupted"]),
            "request_outcome": _result_outcome(rec["http_status"], bool(rec["corrupted"])),
            "http_status": rec["http_status"],
            "corrupted_counter_delta": rec.get("corrupted_delta", ""),
            "prefix_hit_tokens": int(rec.get("prefix_hit_tokens_delta", 0)),
            "input_sha256": rec["input_sha256"],
            "input_note": note,
            "cache_salt": rec.get("cache_salt", ""),
            "output_sha256": rec.get("output_sha256", ""),
        })
    for build, count in seen.items():
        _require_serial_trace_alignment(fbw.get(build, []), count, f"{jsonl}/{build}")
    return rows


COLS = ["cell", "build", "N", "r", "seed", "role", "prompt_tokens",
        "prompt_len_mod_grid", "tail_chunk_len", "tail_step_graph_mode",
        "prefill_chunks", "corrupted", "http_status", "corrupted_counter_delta",
        "prefix_hit_tokens", "input_sha256", "input_note", "cache_salt",
        "output_sha256", "request_outcome"]


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=COLS, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("%s: %d rows" % (os.path.basename(path), len(rows)))


def main():
    os.makedirs(RES, exist_ok=True)
    s7 = step7_rows()
    fr = frozen_rows()
    mn = frozen_rows("mamba_none_requests.jsonl", "mamba_none",
                     "frozen: identical ids; prefix caching OFF, mamba_cache_mode=none")
    write(os.path.join(RES, "step7_requests.csv"), s7)
    if fr:
        write(os.path.join(RES, "frozen_control_requests.csv"), fr)
    if mn:
        write(os.path.join(RES, "mamba_none_requests.csv"), mn)
    fr = fr + mn
    with open(os.path.join(RES, "input_hashes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "build", "seed", "role", "prompt_tokens",
                    "input_sha256", "note"])
        for r in s7 + fr:
            w.writerow([r["cell"], r["build"], r["seed"], r["role"],
                        r["prompt_tokens"], r["input_sha256"], r["input_note"]])
    print("input_hashes.csv written")


def _result_outcome(http_status, corrupted):
    """Keep transport/application failure separate from numerical corruption."""
    if corrupted:
        return "corrupted"
    return "http_error" if http_status >= 400 else "completed"


def _parse_step7_outcome(rest):
    """Parse driver output without treating every HTTP 400 as a NaN incident.

    A non-finite logprob, positive corruption-counter delta, or explicit
    non-finite value in an HTTP error establishes observed corruption. Other
    HTTP failures remain visible as http_error, not as successful requests.
    """
    match = re.search(r"\bHTTP(\d{3})\b", rest)
    http_status = int(match.group(1)) if match else 200
    if not 100 <= http_status <= 599:
        raise ValueError(f"Invalid HTTP status in driver record: {rest!r}")
    delta = re.search(r"\bcorrupted\+(\d+)\b", rest)
    nonfinite = bool(re.search(r"\bnonfinite_lp=True\b", rest))
    explicit_nonfinite_error = http_status >= 400 and bool(re.search(
        r"\b(?:nan|inf|infinity|non[-_ ]?finite)\b", rest, flags=re.IGNORECASE,
    ))
    corrupted = nonfinite or explicit_nonfinite_error or bool(delta and int(delta.group(1)) > 0)
    return http_status, corrupted, _result_outcome(http_status, corrupted)


def _require_serial_trace_alignment(records, request_count, context):
    """Never silently shift request-to-trace joins after a missing trace entry."""
    if len(records) != request_count:
        raise ValueError(
            f"{context}: {request_count} driver requests but {len(records)} trace requests; "
            "cannot join by execution order. Preserve missing data or provide request IDs."
        )


if __name__ == "__main__":
    sys.exit(main())
