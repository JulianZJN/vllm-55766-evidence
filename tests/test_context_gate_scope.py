"""CPU scope probes of #47123's guard, not a reproduction of GDN/NaN.

Bodies are from vLLM 71fc70d3 utils.py/output.py and the published #47123 patch.
Only scheduler containers are synthetic. No CUDA graph is executed.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import pytest

@dataclass
class IterationDetails:
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int
    num_encoder_inputs: int = 0
    num_encoder_output_tokens: int = 0


def compute_iteration_details(scheduler_output):
    num_context_requests = 0
    num_context_tokens = 0
    num_generation_requests = 0
    num_generation_tokens = 0
    new_req_ids = {new_req.req_id for new_req in scheduler_output.scheduled_new_reqs}
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        if scheduler_output.scheduled_cached_reqs.is_context_phase(req_id) or (
            req_id in new_req_ids
        ):
            num_context_requests += 1
            num_context_tokens += num_tokens
        else:
            num_generation_requests += 1
            num_generation_tokens += num_tokens
    scheduled_encoder_input_stats = scheduler_output.scheduled_encoder_input_stats
    num_encoder_inputs = 0
    num_encoder_output_tokens = 0
    if scheduled_encoder_input_stats is not None:
        num_encoder_inputs = scheduled_encoder_input_stats.num_inputs
        num_encoder_output_tokens = scheduled_encoder_input_stats.output_tokens
    return IterationDetails(num_context_requests, num_context_tokens,
                            num_generation_requests, num_generation_tokens,
                            num_encoder_inputs, num_encoder_output_tokens)


class CachedProbe:
    def __init__(self, output_tokens, resumed):
        self._req_id_to_num_output_tokens = {"r": output_tokens}
        self.resumed_req_ids = {"r"} if resumed else set()

    def is_context_phase(self, req_id):
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0


def _compute_force_uniform_decode(scheduler_output, is_hybrid):
    if not is_hybrid:
        return None
    iteration_details = compute_iteration_details(scheduler_output)
    if iteration_details.num_ctx_requests > 0:
        return False
    return None


@pytest.mark.parametrize("hybrid,new,output_tokens,resumed,expected", [
    (False, True, 0, False, None),
    (True, True, 0, False, False),
    (True, False, 0, False, False),
    (True, False, 8, False, None),
    (True, False, 8, True, None),
])
def test_guard_scope(hybrid,new,output_tokens,resumed,expected):
    schedule=SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r")] if new else [],
        scheduled_cached_reqs=CachedProbe(output_tokens,resumed),
        num_scheduled_tokens={"r":6}, scheduled_encoder_input_stats=None)
    assert _compute_force_uniform_decode(schedule,hybrid) is expected
