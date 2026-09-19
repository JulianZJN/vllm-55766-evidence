# Evidence pack — vLLM #55766 / #47123, `r = K+1` prefill tail on a hybrid (GDN) model

Everything referenced from my two comments (issue #55766, PR #47123), plus the
two follow-ups that those comments did not contain: a **frozen-input paired
control** and a **real preemption/resume test** of #47123's context-phase
criterion.

All GPU results in this pack are **SM89 (RTX 4090 D), TP4, one node**. Nothing
here is measured on SM90/H100.

Layout:

| path | what |
|---|---|
| `env/` | versions, GPUs, model revision, full server command lines |
| `patch/` | the diff that was actually applied, and the PR's own diff |
| `scripts/` | every driver used to produce the tables and traces |
| `results/` | per-request CSV/JSONL, one row per request, plus input hashes |
| `traces/` | FBW instrumentation logs (full, gzipped) and trimmed excerpts |
| `tests/` | the CPU-level guard-scope test and its output |
| `MANIFEST.sha256` | sha256 of every file in this pack |

---

## 1. Exact build under test

**vLLM** — two builds, both installed editable into their own venv from a
codeload tarball of the exact commit (the compute nodes have no git):

| build | full SHA | vllm version string | torch |
|---|---|---|---|
| main | `71fc70d3ae1df53300a23ec69f6d97b7207a109f` | `0.29.0.dev0` | `2.13.0+cu130` |
| v0.28.0 | `2cf0a6915ce544dc493a0990f2ea38d81601128a` | `0.28.0` | `2.13.0+cu129` |

All results in `results/` are on the **main** build unless the row says
otherwise. Python 3.12.14. Built with `VLLM_USE_PRECOMPILED=1` against the
matching precompiled wheel commit; both patches under test are Python-only, so
the precompiled kernels are unchanged between the "unfixed" and "fixed" arms.

**GPUs** — 4 × NVIDIA GeForce RTX 4090 D, 24564 MiB each, compute capability
8.9, driver `590.48.01`, CUDA runtime 13.0. All four on one PCIe switch
(`nvidia-smi topo -m`: PIX between every pair), single NUMA node. Tensor
parallel size 4, no pipeline parallelism, no data parallelism. Full output in
`env/gpus.txt`.

## 2. Exact model revision

The weights were fetched from **ModelScope**, repo `Qwen/Qwen3.8-27B`, branch
`master`. ModelScope does not expose a single snapshot id for a branch
download; what the download leaves on disk is:

```
.mv   -> Revision:master,CreatedAt:1786731894      (2026-08-14T18:24:54Z)
.msc  -> per-file blob revisions; the 18 safetensors shards, config.json,
         merges.txt, crc32.txt ... all carry
         1098534ab5d7220ea0f4a6b9f07bb03729a79c1d
         configuration.json carries
         7005cc891d46d4db578d99171526465d50f69b15
```

so the snapshot is `master` as of `2026-08-14T18:24:54Z`, file revision
`1098534ab5d7220ea0f4a6b9f07bb03729a79c1d`. Local download completed
2026-09-19 01:51 (+0800), 18 shards.

Because a branch name is not a stable identifier, identify the weights by these
hashes instead:

| file | sha256 |
|---|---|
| `config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `model.safetensors.index.json` | `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` |
| `generation_config.json` | `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e` |
| `tokenizer_config.json` | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |

`config.json`: `architectures: ["Qwen3_5ForConditionalGeneration"]`,
`model_type: qwen3_5`, `dtype: bfloat16`, `full_attention_interval: 4`
(`layer_types` = three `linear_attention` layers per `full_attention` layer),
`hidden_size: 5120`, `head_dim: 256`. Served with `--dtype bfloat16`.

## 3. Resolved geometry (printed by the engine at startup)

```
GDN_DBG GEOM cache_config_block_size=816 hash_block_size=816
             mamba_state_block_size=816 mamba_cache_mode=align need_split=True
             groups=[('MambaSpec',816), ('MambaSpec',816), ('MambaSpec',816),
                     ('FullAttentionSpec',816)]
GPU KV cache size: 168,068 tokens
```

`num_speculative_tokens = 5`, so `K + 1 = 6` and the align grid is **816**.
(The grid depends on the spec config: `num_spec=1` resolves to 800, ngram off
to 784 — the `GDN_DBG GEOM` line in each trace states which was in force.)

## 4. Server command lines

`env/server_cmdlines.txt` has them verbatim, with the home directory replaced
by `<MODEL_DIR>`. The baseline (all `r`-sweep and frozen-control runs):

```
python -m vllm.entrypoints.openai.api_server \
  --model <MODEL_DIR>/Qwen__Qwen3.8-27B --served-model-name qwen38 \
  --host 127.0.0.1 --port <PORT> --tensor-parallel-size 4 --trust-remote-code \
  --dtype bfloat16 --enable-prefix-caching --gpu-memory-utilization 0.88 \
  --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 64 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,
                         "prompt_lookup_max":5,"prompt_lookup_min":2}' \
  --mamba-cache-mode align --no-enable-log-requests
```

with `VLLM_COMPUTE_NANS_IN_LOGITS=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`,
`VLLM_DEBUG_FBW=<1|2>`, `VLLM_DEBUG_GDN_PREFIX_STATE=1`. cudagraph mode is the
engine default for this model, `FULL_AND_PIECEWISE`.

Variants: the preemption runs add `--num-gpu-blocks-override 84
--max-num-seqs 8`; the `mamba_cache_mode=none` cell adds
`--no-enable-prefix-caching` (see §6).

## 5. What of #47123 was applied

PR #47123 at the head I read, `41edfaf77e3b4765a43d58f7cd81445aa481b83d`,
touches exactly two files:

* `vllm/v1/worker/gpu_model_runner.py` — the runner change
* `tests/v1/worker/test_gpu_model_runner.py` — `test_compute_force_uniform_decode`

`patch/pr_47123_reference.diff`
(sha256 `b577ab88c4983d148ccda6a19030200fc81453b42f0deabbe8a83439b1f5c3a5`)
is that PR diff, both files.

The "fixed" arm is **main `71fc70d3` plus only the four
`gpu_model_runner.py` hunks of that diff** — not the PR head, and not a merge
of the PR branch. The applied diff (as produced on the running worktree, which
also carries the env-guarded FBW hooks, hence the extra import line) is
`patch/pr47123_applied_to_71fc70d3.diff`, sha256
`36aee0d16a0324377c490b9ff4e79d597a034d275139d71e4f3a3f525244b65c`. The PR's
unit test was applied separately, run, and reverted; it is not part of the
serving build.

Applying the hunks to `71fc70d3` needed offsets (`-3`, `-59`, `-62`, `-62`
lines) and fuzz 2 on the import hunk; the resulting function body is identical
to the PR's.

## 6. `mamba_cache_mode` is not independently selectable (correction)

`vllm/model_executor/models/config.py:620-657` (`MambaModelConfig.
verify_and_update_config`) forces `mamba_cache_mode = "align"` whenever prefix
caching is enabled, and forces it to `"none"` whenever prefix caching is
disabled. Passing `--mamba-cache-mode none` together with
`--enable-prefix-caching` is silently a no-op: the log says *"Mamba cache mode
is set to 'align' … by default when prefix caching is enabled"* and the
`GDN_DBG GEOM` line still reports `mamba_cache_mode=align need_split=True`.

So an earlier note of mine — that the failure is "independent of
`mamba_cache_mode`" and that "none is worse" — was based on a run that was
still in align mode. A genuine `none` run (prefix caching off, `GDN_DBG GEOM …
mamba_cache_mode=none need_split=False mamba_state_block_size=32768`) is in
`results/mamba_none_requests.csv`; its prefill chunking is 8192 / 8134 with no
6-token tail, and 0/9 requests corrupt. Mamba cache mode and prefix caching are
confounded on this model, so no claim of independence is available.

## 7. How to read `results/`

`results/step7_requests.csv`, `results/frozen_control_requests.csv`,
`results/mamba_none_requests.csv` and `results/preempt_requests.jsonl` are one
row per request. Columns:

| column | meaning |
|---|---|
| `cell` | the cell this row belongs to; **never merge cells** |
| `build` | `unfixed` = main `71fc70d3`; `fixed` = + the #47123 runner hunks |
| `N`, `r` | prompt was `N*816 + r` tokens |
| `seed` | driver seed for the deterministic token-id generator |
| `role` | `A`, `B` (prefix-cache restore of A) or `B-salted` (fresh `cache_salt`) |
| `prompt_tokens` | length of the token-id list actually sent |
| `prompt_len_mod_grid` | `prompt_tokens % 816` |
| `tail_chunk_len` | tokens in the request's **last prefill chunk** (from the trace) |
| `tail_step_graph_mode` | runtime cudagraph mode of that step: `FULL`/`PIECEWISE`/`NONE` |
| `prefill_chunks` | the whole chunk sequence, `len:mode` |
| `corrupted` | 1 iff `vllm:corrupted_requests_total` moved, or HTTP 400 "…nan…", or a non-finite logprob |
| `http_status` | 200, or 400 when a NaN logprob made the response non-JSON-encodable |
| `prefix_hit_tokens` | `vllm:prefix_cache_hits` delta for this request |
| `input_sha256` | sha256 of the request's input token-id list (decimal, comma-joined) |
| `input_note` | whether that input was frozen across arms |

`results/input_hashes.csv` is the same hashes on their own, so two runs can be
checked for identical inputs without reading the tables.

**The step-7 rows are not a frozen-input comparison.** In those runs B was
built as `A's prompt ids + the reply that that arm's server produced for A +
suffix`. On the unfixed arm A returned HTTP 400 for the failing seeds, so B was
sent 18326 tokens there against 18626 on the fixed arm — visible directly in
`step7_requests.csv`. That is why `frozen_control_requests.csv` exists: there
the reply is a fixed synthetic continuation, so every arm receives byte-identical
token-id lists (same `input_sha256`).

## 8. What is in `traces/`

| path | what |
|---|---|
| `fbw_first_bad_write_trimmed.txt` | the steps that show the first bad write (steps 3 / 54 / 55 / 57 / 97) |
| `fbw_piecewise_control_trimmed.txt` | the same steps under `cudagraph_mode=PIECEWISE`, and under main + the #47123 hunks |
| `fbw/` | the full gzipped FBW logs those excerpts come from, plus the workload logs |
| `step7/` | the r-sweep cells: one FBW log and one per-request log per (build, r, N) |
| `frozen/` | FBW logs of the three frozen-input arms (unfixed, fixed, unfixed replicate) |
| `mamba_none/` | FBW log of the true `mamba_cache_mode=none` cell |
| `preempt/` | preemption runs: full log, per-resume-episode summary, and the trimmed K+1 case |
| `driver_stdout/` | raw stdout of every driver run, one line per request |

`preempt/resume_episodes.txt` is `scripts/preempt_analyze.py` over
`preempt/fbw_preempt_meta_only.log.gz`: 24 resume episodes, each step of each
replay with its scheduled token count, runtime cudagraph mode, the resumed
request's `num_output_tokens`, whether `is_context_phase` (or "is a new request")
holds for it, the step's `num_ctx_requests`, and the live return value of
#47123's `_compute_force_uniform_decode`.

`preempt/fbw_preempt_full_trace.log.gz` + `resume_episodes_full_trace.txt` are
four reruns of the same configuration with the full `VLLM_DEBUG_FBW=1`
instrumentation (state checksums on every step): 18 preemptions, 20 resume
episodes, no K+1 tail among them. The instrumentation slows each step and moves
the preemption point, and the tail length is decided by the residue at
preemption — so there is no state-checksum trace of a resumed K+1 tail, only the
dispatch evidence in the META-only log.

`preempt/k_plus_1_resumed_tail_trimmed.txt` is the pair that matters: the only
two 6-token recompute tails in those runs. Same step shape
(`ntok=12/12 nreq=2/2 sched=[6,6] k=[5,0] npf=1 nspec=1`), opposite outcome,
with `num_output_tokens` of the resumed request the only difference.

## 9. Reproducing

`scripts/serve_followup.sh` starts the server (knobs documented in its header).
Then, for the frozen control:

```
python scripts/frozen_ab.py --base http://127.0.0.1:8140 --N 20 --r 6 \
    --seeds 3 --build unfixed --cell B-frozen-r6-N20 --out frozen.jsonl
```

and for the preemption test:

```
python scripts/preempt_probe.py --base http://127.0.0.1:8141 \
    --nfillers 4 --filler-blocks 8 --filler-max 2400 \
    --victim-blocks 8 --victim-offset 0 --victim-max 3000 \
    --label calib --out preempt.jsonl
python scripts/preempt_analyze.py <fbw log>
```

`scripts/inject_fbw2.py <worktree>` adds the instrumentation (env-guarded by
`VLLM_DEBUG_FBW`; `--revert` removes it). `scripts/build_tables.py` regenerates
everything in `results/` from `traces/`.
