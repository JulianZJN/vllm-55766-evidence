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
    explicit_nonfinite_error = http_status >= 400 and bool(
        re.search(
            r"\b(?:nan|inf|infinity|non[-_ ]?finite)\b",
            rest,
            flags=re.IGNORECASE,
        )
    )
    corrupted = (
        nonfinite
        or explicit_nonfinite_error
        or bool(delta and int(delta.group(1)) > 0)
    )
    return http_status, corrupted, _result_outcome(http_status, corrupted)


def _require_serial_trace_alignment(records, request_count, context):
    """Never silently shift request-to-trace joins after a missing trace entry."""
    if len(records) != request_count:
        raise ValueError(
            f"{context}: {request_count} driver requests but {len(records)} trace requests; "
            "cannot join by execution order. Preserve missing data or provide request IDs."
        )
