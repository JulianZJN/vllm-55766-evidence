#!/usr/bin/env python3
"""Summarise the resume episodes in an FBW log (work item C.2).

usage: preempt_analyze.py <fbw log (.gz or plain)>

For every step whose META line marks a request as resumed-from-preemption,
follow that request until its recompute finishes (the step whose scheduled
token count ends the replay) and print, per step:

  sched      tokens scheduled for the resumed request this step
  mode       runtime cudagraph mode of the step
  uniform    the batch_descriptor's uniform flag
  nout       num_output_tokens the scheduler reported for it
  ctx        is_context_phase(req) or req is a scheduled_new_req
  nctx       compute_iteration_details(...).num_ctx_requests for the step
  fud        what #47123's _compute_force_uniform_decode returned

The line to look for is a recompute tail of exactly 1+num_spec tokens whose
step still runs mode=FULL.
"""
import ast
import gzip
import re
import sys

PAT = re.compile(r"FBW step=(\d+) META ")


def fields(line):
    d = {}
    d["mode"] = re.search(r"mode=(\w+)", line).group(1)
    d["uniform"] = re.search(r"uniform=(\w+)", line).group(1)
    head = line[:line.find("owned=")] if "owned=" in line else line
    d["reqs"] = re.findall(r"'([0-9a-f]{6})'", head)
    d["sched"] = ast.literal_eval(re.search(r"sched=(\[[^\]]*\])", line).group(1))
    d["k"] = ast.literal_eval(re.search(r" k=(\[[^\]]*\])", line).group(1))
    for f in ("resumed", "nout", "ctx", "ncomp"):
        m = re.search(r"\b%s=(\[[^\]]*\])" % f, line)
        d[f] = m.group(1) if m else "?"
    for f in ("nctx", "fud", "npf"):
        m = re.search(r"\b%s=([^ ]+)" % f, line)
        d[f] = m.group(1) if m else "?"
    return d


def main(path):
    op = gzip.open if path.endswith(".gz") else open
    steps = []
    with op(path, "rt", errors="replace") as fh:
        for line in fh:
            i = line.find("FBW step=")
            if i < 0:
                continue
            line = line[i:]
            m = PAT.match(line)
            if m:
                steps.append((int(m.group(1)), fields(line)))
    idx = {s: i for i, (s, _) in enumerate(steps)}
    episodes = 0
    for i, (s, d) in enumerate(steps):
        res = ast.literal_eval(d["resumed"]) if d["resumed"].startswith("[") else []
        for rid in res:
            episodes += 1
            print("\n=== resume episode: req=%s first step=%d" % (rid, s))
            for j in range(i, min(i + 60, len(steps))):
                s2, d2 = steps[j]
                if rid not in d2["reqs"]:
                    break
                p = d2["reqs"].index(rid)
                sched = d2["sched"][p] if p < len(d2["sched"]) else "?"
                k = d2["k"][p] if p < len(d2["k"]) else "?"
                nout = ast.literal_eval(d2["nout"]) if d2["nout"].startswith("[") else []
                ctx = ast.literal_eval(d2["ctx"]) if d2["ctx"].startswith("[") else []
                ncomp = (ast.literal_eval(d2["ncomp"])
                         if d2["ncomp"].startswith("[") else [])
                print("  step=%-6d sched=%-6s k=%-2s mode=%-10s uniform=%-5s "
                      "nout=%-6s ctx=%-5s nctx=%-3s fud=%-5s ncomp=%-7s nreq=%d"
                      % (s2, sched, k, d2["mode"], d2["uniform"],
                         nout[p] if p < len(nout) else "?",
                         ctx[p] if p < len(ctx) else "?",
                         d2["nctx"], d2["fud"],
                         ncomp[p] if p < len(ncomp) else "?", len(d2["reqs"])))
                if j > i and sched == 1:
                    break
                if j > i and isinstance(sched, int) and sched <= 6 and k == 5:
                    break
    print("\nresume episodes: %d" % episodes)


if __name__ == "__main__":
    main(sys.argv[1])
