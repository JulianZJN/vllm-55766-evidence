"""Synthetic parser checks, not reproductions of the published GPU runs."""

import pytest

from scripts.build_tables import (
    _parse_step7_outcome,
    _require_serial_trace_alignment,
)


@pytest.mark.parametrize(
    "line,http,corrupt,outcome",
    [
        (
            "corrupted+0 prefix_hit_tokens+0 nonfinite_lp=False",
            200,
            False,
            "completed",
        ),
        ("HTTP400 bad request: invalid model", 400, False, "http_error"),
        ("HTTP500 internal server error", 500, False, "http_error"),
        ("HTTP503 unavailable corrupted+0", 503, False, "http_error"),
        (
            "HTTP400 Out of range float values are not JSON compliant: nan",
            400,
            True,
            "corrupted",
        ),
        ("HTTP400 non-finite logprob", 400, True, "corrupted"),
        ("HTTP400 Infinity value", 400, True, "corrupted"),
        ("corrupted+1 nonfinite_lp=False", 200, True, "corrupted"),
        ("corrupted+0 nonfinite_lp=True", 200, True, "corrupted"),
        ("HTTP400 cannot parse banana", 400, False, "http_error"),
        ("HTTP500 corrupted+2", 500, True, "corrupted"),
    ],
)
def test_outcome(line, http, corrupt, outcome):
    assert _parse_step7_outcome(line) == (http, corrupt, outcome)


def test_rejects_invalid_http_code():
    with pytest.raises(ValueError):
        _parse_step7_outcome("HTTP999")


@pytest.mark.parametrize("trace_count,request_count", [(2, 3), (3, 2), (0, 3)])
def test_missing_trace_cannot_silently_shift_rows(trace_count, request_count):
    with pytest.raises(ValueError, match="cannot join by execution order"):
        _require_serial_trace_alignment(
            [{}] * trace_count, request_count, "fixture"
        )


def test_matching_counts_are_only_an_alignment_precondition():
    _require_serial_trace_alignment([{}] * 3, 3, "fixture")
