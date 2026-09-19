#!/usr/bin/env python3
"""Analyze an FBW server log for step7: per request (in order) report prefill chunking, FULL-mode
prefill-tail steps, stray-written blocks on that step, and whether those blocks are later shared
(prefix-cache restored) by the next requests. usage: step7_analyze.py <fbw_log(.gz)>"""
import ast, gzip, re, sys

def lines(p):
    f = gzip.open(p, "rt") if p.endswith(".gz") else open(p, errors="replace")
    for l in f:
        i = l.find("FBW step=")
        if i >= 0:
            yield l[i:].rstrip("\n")

def blocks(field, l):
    m = re.search(field + r"=\[([^\]]*)\]", l)
    if not m or not m.group(1):
        return set()
    return {int(x.split(":")[0]) for x in m.group(1).split(",") if x and x[0].isdigit()}

steps = {}
for l in lines(sys.argv[1]):
    m = re.match(r"FBW step=(\d+) (P1|P2|P3|META) ", l)
    if not m:
        continue
    s = int(m.group(1)); d = steps.setdefault(s, {})
    if m.group(2) == "META":
        d["mode"] = re.search(r"mode=(\w+)", l).group(1)
        d["req"] = re.search(r"reqs=\['([^']*)'", l).group(1)
        d["sched"] = int(re.search(r"sched=\[(\d+)", l).group(1))
        d["k"] = int(re.search(r" k=\[(\d+)", l).group(1))
        d["npf"] = int(re.search(r"npf=(\d+)", l).group(1))
        om = re.search(r"owned=(\{.*?\}\}) gid=", l)
        d["owned"] = list(ast.literal_eval(om.group(1)).values())[0]
        d["nsi"] = re.search(r"nsi=(\[[^\]]*\]|None)", l).group(1)
        d["ssi"] = re.search(r"ssi=(\[\[.*?\]\]|None)", l).group(1)
    else:
        ph = m.group(2)
        d[ph] = (blocks("chg_conv", l), blocks("chg_ssm", l))
        am = re.search(r"attn_view_nonfinite_blocks=\[([^\]]*)\]", l)
        d[ph + "_attnbad"] = {int(x) for x in am.group(1).split(",")} if am and am.group(1).strip() else set()

reqs = []
for s in sorted(steps):
    d = steps[s]
    if "req" not in d:
        continue
    if not reqs or reqs[-1]["id"] != d["req"]:
        reqs.append({"id": d["req"], "steps": []})
    reqs[-1]["steps"].append(s)

names = []
for i, _ in enumerate(reqs):
    names.append("seed%d-%s" % (i // 3, ["A", "B", "Bsalt"][i % 3]))

prev_owned_final = None
for i, rq in enumerate(reqs):
    st = [steps[s] for s in rq["steps"]]
    pf = [(s, steps[s]) for s in rq["steps"] if steps[s]["npf"] > 0]
    chunks = [(d["sched"], d["mode"]) for _, d in pf]
    first = st[0]; last = st[-1]
    g3 = [b for b in first["owned"][3] if b]
    print("== %s req=%s prefill_chunks=%s first_step=%d" % (names[i], rq["id"], chunks, rq["steps"][0]))
    # prefix-cache sharing with the previous requests: same block id at same position at first step
    if i % 3 != 0:
        a_final = reqs[i - (i % 3)]["final_owned"]
        shared = {g: [b for p, b in enumerate(first["owned"][g]) if b and p < len(a_final[g]) and a_final[g][p] == b]
                  for g in first["owned"]}
        print("   blocks shared with A at B's first step (same id, same position):", shared)
        bad = sorted(set(g3) & first.get("P1_attnbad", set()))
        print("   own attention blocks already non-finite BEFORE first forward:", bad)
    for s, d in pf:
        if d["mode"] == "FULL":
            conv, ssm = d.get("P2", (set(), set()))
            own = d["owned"]
            pos = None
            try:
                nsi = ast.literal_eval(d["nsi"])[0]
                pos = max(p for p, b in enumerate(own[0]) if b == nsi)
            except Exception:
                nsi = None
            intended = {own[g][pos] for g in (0, 1, 2)} if pos is not None else set()
            cur_attn = {b for b in own[3] if b}
            legit_attn = [b for b in own[3] if b][-1:]  # block receiving this chunk's KV
            stray = (conv | ssm) - intended - set(legit_attn)
            stray_attn = sorted(stray & cur_attn)
            attn_new_bad = sorted((d.get("P2_attnbad", set()) - d.get("P1_attnbad", set())) & cur_attn)
            print("   FULL-mode prefill tail: step=%d sched=%d intended(gid0-2)=%s" % (s, d["sched"], sorted(intended)))
            print("     changed conv=%s ssm=%s" % (sorted(conv), sorted(ssm)))
            print("     stray (changed minus intended)=%s" % sorted(stray))
            print("     stray blocks that are THIS request's attention blocks=%s (position in gid3 list: %s)"
                  % (stray_attn, [own[3].index(b) for b in stray_attn]))
            print("     own attention blocks that turned non-finite in this step=%s" % attn_new_bad)
            rq["stray"] = stray
    rq["final_owned"] = last["owned"]
    if i % 3 != 0:
        a = reqs[i - (i % 3)]
        if "stray" in a:
            cur = {b for g in first["owned"] for b in first["owned"][g] if b}
            sh = {b for g in shared for b in shared[g]}
            print("   A's stray-written blocks that this request restored from cache:", sorted(a["stray"] & sh))
