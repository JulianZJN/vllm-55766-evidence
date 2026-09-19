#!/usr/bin/env python3
"""First-bad-write instrumentation injector (env-guarded by VLLM_DEBUG_FBW=1).

usage: inject_fbw.py <worktree> [--revert]

Adds vllm/v1/worker/_fbw_debug.py and three guarded hook lines to
vllm/v1/worker/gpu_model_runner.py (backup: gpu_model_runner.py.fbw_orig).
With VLLM_DEBUG_FBW unset the hooks are a single module-attribute test.
All GPU reads happen in the runner, OUTSIDE any captured graph, after an
explicit torch.cuda.synchronize(). Only TP rank 0 logs.

Phases logged per model step (layer 0 = first GDN layer, whole block pool):
  P1 = just before forward (diff vs previous P3: align pre-copy etc.)
  P2 = just after forward  (diff vs P1: what the forward/graph replay wrote)
  P3 = after sampling + _update_states_after_model_execute (postprocess copy)
"""
import os
import shutil
import sys

MODULE = r'''
# SPDX-License-Identifier: Apache-2.0
# Debug-only: first-bad-write tracer. Active iff VLLM_DEBUG_FBW=1.
import os
import re
import sys

import torch

ON = os.environ.get("VLLM_DEBUG_FBW") == "1"
_S = {"step": 0, "prev": None, "layer": None, "gid": None, "rank": None}
_CAP = 48


def _rank():
    if _S["rank"] is None:
        from vllm.distributed import get_tensor_model_parallel_rank

        _S["rank"] = get_tensor_model_parallel_rank()
    return _S["rank"]


def _log(msg):
    sys.stderr.write("FBW " + msg + "\n")
    sys.stderr.flush()


def _find_layer(runner):
    if _S["layer"] is not None:
        return
    from vllm.v1.kv_cache_interface import MambaSpec

    best = None
    for gid, g in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not isinstance(g.kv_cache_spec, MambaSpec):
            continue
        for name in g.layer_names:
            m = re.search(r"layers\.(\d+)\.", name)
            idx = int(m.group(1)) if m else 10**9
            if best is None or idx < best[0]:
                best = (idx, name, gid)
    assert best is not None
    _S["layer"], _S["gid"] = best[1], best[2]
    kv = _kv(runner)
    _log(
        "INIT layer=%s gid=%d groups=%s conv=%s ssm=%s"
        % (
            best[1],
            best[2],
            [type(g.kv_cache_spec).__name__ for g in
             runner.kv_cache_config.kv_cache_groups],
            tuple(kv[0].shape),
            tuple(kv[1].shape),
        )
    )


def _kv(runner):
    kv = runner.compilation_config.static_forward_context[_S["layer"]].kv_cache
    if isinstance(kv[0], (list, tuple)):
        kv = kv[0]
    return kv


def _sums(t):
    out = []
    for i in range(0, t.shape[0], 32):
        c = t[i : i + 32]
        out.append(c.abs().sum(dim=tuple(range(1, c.dim())), dtype=torch.float64))
    return torch.cat(out)


def _attn_nonfinite(runner, nblk):
    """Per-block non-finite scan of the first full-attention layer's KV cache
    (its KVCacheTensor is shared with the first GDN layers in hybrid models)."""
    if _S.get("attn") is None:
        from vllm.v1.kv_cache_interface import MambaSpec

        for g in runner.kv_cache_config.kv_cache_groups:
            if not isinstance(g.kv_cache_spec, MambaSpec):
                names = sorted(g.layer_names, key=lambda n: int(
                    re.search(r"layers\.(\d+)\.", n).group(1)))
                _S["attn"] = names[0]
                break
        t = runner.compilation_config.static_forward_context[_S["attn"]].kv_cache
        if isinstance(t, (list, tuple)):
            t = t[0]
        _log("INIT attn_layer=%s kv_shape=%s dtype=%s gdn_dtypes=%s"
             % (_S["attn"], tuple(t.shape), t.dtype,
                [x.dtype for x in _kv(runner)]))
    t = runner.compilation_config.static_forward_context[_S["attn"]].kv_cache
    if isinstance(t, (list, tuple)):
        t = t[0]
    dim = list(t.shape).index(nblk)
    t = t.movedim(dim, 0)
    bad = []
    for i in range(0, nblk, 32):
        c = t[i : i + 32]
        nf = (~torch.isfinite(c)).reshape(c.shape[0], -1).any(dim=1)
        bad += [i + j for j in nf.nonzero().flatten().tolist()]
    return bad


def _snap(runner):
    torch.cuda.synchronize()
    kv = _kv(runner)
    conv, ssm = _sums(kv[0]).cpu(), _sums(kv[1]).cpu()
    try:
        _S["attn_bad"] = _attn_nonfinite(runner, conv.numel())
    except Exception as e:  # debug-only; never break the step
        _S["attn_bad"] = "ERR %r" % (e,)
    torch.cuda.synchronize()
    return conv, ssm


def _diff(cur, prev):
    same = (cur == prev) | (torch.isnan(cur) & torch.isnan(prev))
    return (~same).nonzero().flatten().tolist()


def _fmt(idx, cur):
    s = ["%d:%.6g" % (b, cur[b].item()) for b in idx[:_CAP]]
    if len(idx) > _CAP:
        s.append("...+%d" % (len(idx) - _CAP))
    return "[" + ",".join(s) + "]"


def _phase(runner, tag):
    conv, ssm = _snap(runner)
    prev = _S["prev"]
    if prev is None:
        nf_c = (~torch.isfinite(conv)).nonzero().flatten().tolist()
        nf_s = (~torch.isfinite(ssm)).nonzero().flatten().tolist()
        _log("step=%d %s FIRST nblk=%d nonfinite_conv=%s nonfinite_ssm=%s"
             % (_S["step"], tag, conv.numel(), nf_c, nf_s))
    else:
        dc, ds = _diff(conv, prev[0]), _diff(ssm, prev[1])
        nf_c = (~torch.isfinite(conv)).nonzero().flatten().tolist()
        nf_s = (~torch.isfinite(ssm)).nonzero().flatten().tolist()
        _log("step=%d %s chg_conv=%s chg_ssm=%s nonfinite_conv=%s nonfinite_ssm=%s "
             "attn_view_nonfinite_blocks=%s"
             % (_S["step"], tag, _fmt(dc, conv), _fmt(ds, ssm),
                nf_c[:_CAP], nf_s[:_CAP], _S.get("attn_bad")))
    _S["prev"] = (conv, ssm)
    return conv, ssm


def _tl(t, n=None):
    if t is None:
        return None
    t = t.detach()
    if n is not None:
        t = t[:n]
    return t.cpu().tolist()


def pre(runner, scheduler_output, attn_metadata, cudagraph_mode, batch_desc,
        num_reqs, num_reqs_padded, ntok, ntok_padded):
    if _rank() != 0:
        return
    _find_layer(runner)
    _S["step"] += 1
    conv, ssm = _phase(runner, "P1")
    ib = runner.input_batch
    req_ids = list(ib.req_ids[:num_reqs])
    sched = [scheduler_output.num_scheduled_tokens[r] for r in req_ids]
    k = [len(scheduler_output.scheduled_spec_decode_tokens.get(r, ()))
         for r in req_ids]
    owned = {}
    for r in req_ids:
        bids = runner.requests[r].block_ids
        owned[r[-6:]] = {g: list(bids[g]) for g in range(len(bids))}
    md = attn_metadata
    if isinstance(md, list):
        md = md[0]
    m = md.get(_S["layer"]) if isinstance(md, dict) else None
    meta = "none"
    if m is not None:
        ssi = _tl(m.spec_state_indices_tensor)
        nacc = _tl(m.num_accepted_tokens)
        init = None
        if ssi is not None and nacc is not None:
            init = []
            for row, a in zip(ssi, nacc):
                col = max(0, min(a - 1, len(row) - 1))
                b = row[col]
                init.append((b, a, bool(torch.isfinite(ssm[b])),
                             bool(torch.isfinite(conv[row[0]]))))
        meta = (
            "npf=%d ndec=%d nspec=%d nspectok=%d nactual=%d ssi=%s nsi=%s nacc=%s "
            "sqsl=%s nsqsl=%s mask=%s init(blk,nacc,ssm_finite,conv_finite)=%s"
            % (m.num_prefills, m.num_decodes, m.num_spec_decodes,
               m.num_spec_decode_tokens, m.num_actual_tokens, ssi,
               _tl(m.non_spec_state_indices_tensor, 8), nacc,
               _tl(m.spec_query_start_loc), _tl(m.non_spec_query_start_loc, 9),
               _tl(m.spec_sequence_masks), init)
        )
    _log(
        "step=%d META mode=%s uniform=%s ntok=%d/%d nreq=%d/%d reqs=%s sched=%s k=%s "
        "nacc_cpu=%s owned=%s gid=%d %s"
        % (_S["step"], cudagraph_mode.name,
           getattr(batch_desc, "uniform", None), ntok, ntok_padded, num_reqs,
           num_reqs_padded, [r[-6:] for r in req_ids], sched, k,
           ib.num_accepted_tokens_cpu[:num_reqs].tolist(), owned, _S["gid"], meta)
    )


def post(runner):
    if _rank() != 0 or _S["layer"] is None:
        return
    _phase(runner, "P2")


def post_sample(runner):
    if _rank() != 0 or _S["layer"] is None:
        return
    _phase(runner, "P3")
'''

A1 = "        is_padding = self._prepare_padding_mask(num_tokens_unpadded, num_tokens_padded)\n"
H1 = (
    "        if _fbw_debug.ON:  # FBW-DEBUG\n"
    "            _fbw_debug.pre(self, scheduler_output, attn_metadata, cudagraph_mode, batch_desc, num_reqs, num_reqs_padded, num_tokens_unpadded, num_tokens_padded)  # FBW-DEBUG\n"
)
A2 = '        with record_function_or_nullcontext("gpu_model_runner: postprocess"):\n'
H2 = (
    "        if _fbw_debug.ON:  # FBW-DEBUG\n"
    "            _fbw_debug.post(self)  # FBW-DEBUG\n"
)
A3 = (
    "        self._update_states_after_model_execute(\n"
    "            sampler_output.sampled_token_ids, scheduler_output\n"
    "        )\n"
)
H3 = (
    "        if _fbw_debug.ON:  # FBW-DEBUG\n"
    "            _fbw_debug.post_sample(self)  # FBW-DEBUG\n"
)
AI = "from vllm.v1.worker import mamba_utils\n"
HI = "from vllm.v1.worker import _fbw_debug  # FBW-DEBUG\n"


def main():
    wt = sys.argv[1]
    f = os.path.join(wt, "vllm/v1/worker/gpu_model_runner.py")
    mod = os.path.join(wt, "vllm/v1/worker/_fbw_debug.py")
    bak = f + ".fbw_orig"
    if "--revert" in sys.argv:
        if os.path.exists(bak):
            shutil.copyfile(bak, f)
            os.remove(bak)
        if os.path.exists(mod):
            os.remove(mod)
        print("reverted")
        return
    src = open(f).read()
    if "FBW-DEBUG" in src:
        print("already injected; rewriting module only")
    else:
        shutil.copyfile(f, bak)
        for a in (A1, A2, A3, AI):
            assert src.count(a) == 1, a
        src = src.replace(A1, H1 + A1).replace(A2, H2 + A2)
        src = src.replace(A3, A3 + H3).replace(AI, AI + HI)
        open(f, "w").write(src)
    open(mod, "w").write(MODULE.lstrip("\n"))
    print("injected")


if __name__ == "__main__":
    main()
