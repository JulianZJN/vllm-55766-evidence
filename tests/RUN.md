# CPU guard-scope test

`test_context_gate_scope.py` is a CPU-level check of the guard predicate. It re-executes
the bodies of `compute_iteration_details` (vLLM 71fc70d3 `v1/utils.py`),
`CachedRequestData.is_context_phase` (`v1/core/sched/output.py:182-184`) and
#47123's `_compute_force_uniform_decode` against synthetic scheduler
containers. No CUDA graph runs; this is a scope check of the predicate, not a
reproduction of anything.

Run for the record in the same venv that serves the model
(Python 3.12.14, vLLM 0.29.0.dev0 @ 71fc70d3):

```
$ python -m pytest -q test_context_gate_scope.py
.....                                                                    [100%]
5 passed in 0.01s
```

The five cases and what they say:

| hybrid | new req | num_output_tokens | resumed | `_compute_force_uniform_decode` |
|---|---|---|---|---|
| no | yes | 0 | no | `None` (non-hybrid defers to the heuristic) |
| yes | yes | 0 | no | `False` (new request is context phase) |
| yes | no | 0 | no | `False` (chunked prefill, still context phase) |
| yes | no | 8 | no | `None` |
| yes | no | 8 | **yes** | `None` |

The last row is the one that matters: a cached request carrying
`resumed_req_ids` but `num_output_tokens > 0` is **not** forced off the uniform
path. That is a statement about the predicate only. What a real preempted and
resumed request actually gets scheduled as, and which cudagraph mode its
recompute chunks run in, is in `../results/preempt_requests.jsonl` and
`../traces/preempt/`.
