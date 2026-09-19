# results/

One row per request. Denominators are kept per cell; nothing here is averaged
or merged across `r`, `N`, build or role.

| file | cells |
|---|---|
| `step7_requests.csv` | the published r-sweep, 126 requests: r ∈ {4,6,7,8,10}, N ∈ {20,34}, builds `unfixed` / `fixed`, roles A / B / Bsalt, 3 seeds each |
| `frozen_control_requests.csv` / `.jsonl` | the frozen-input paired control: r=6, N=20, 3 seeds, roles A / B / B-salted, builds `unfixed`, `fixed`, and `unfixed-replicate` (a second independent server, same config) |
| `mamba_none_requests.csv` / `.jsonl` | the true `mamba_cache_mode=none` cell (prefix caching off), r=6, N=20, 3 seeds |
| `preempt_requests.jsonl` | the preemption/resume runs: fillers + victim per run, plus a SUMMARY row per run carrying `vllm:num_preemptions_total` before/after |
| `input_hashes.csv` | sha256 of each request's input token-id list, with a note saying whether that input was frozen across arms |

## Counts, per cell, so they can be checked against the tables

frozen control, r=6, N=20, corrupt requests out of 3 seeds:

| build | A | B (prefix hit) | B-salted | tail chunk | tail step graph mode |
|---|---|---|---|---|---|
| unfixed | 2/3 | 2/3 | 0/3 | 6 | FULL |
| unfixed-replicate | 2/3 | 2/3 | 0/3 | 6 | FULL |
| fixed | 0/3 | 0/3 | 0/3 | 6 | PIECEWISE |

All 27 rows carry `prompt_tokens` 16326 (A) / 18626 (B, B-salted) and the same
nine `input_sha256` values across the three builds.

true `mamba_cache_mode=none`, r=6, N=20: 0/3 A, 0/3 B, 0/3 B-salted. Prefill
chunking 8192 / 8134 (A) and 8192 / 8192 / 2242 (B): no align residue, so no
6-token tail exists and no prefill step runs FULL.

`corrupted` is 1 iff any of: `vllm:corrupted_requests_total` moved for that
request, the response was HTTP 400 "Out of range float values are not JSON
compliant: nan" (which is what a NaN logprob produces), or a returned logprob
was non-finite.
