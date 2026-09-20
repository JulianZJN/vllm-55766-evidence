#!/usr/bin/env python3
"""Harden serial evidence parsing without regenerating published result tables."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = "scripts/build_tables.py"
EXPECTED = "8fa10822780f0bd142f7cf2f492cbab16c8148f0"


def once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one source anchor, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    text = once(
        text,
        "        steps[s] = d",
        '''        if len(d["reqs"]) > 1:
            raise ValueError(f"{path}: step {s} has multiple requests; serial parser cannot attribute rows")
        if len(d["sched"]) != len(d["reqs"]) or len(d["k"]) != len(d["reqs"]):
            raise ValueError(f"{path}: inconsistent per-request metadata at step {s}")
        if s in steps and steps[s] != d:
            raise ValueError(f"{path}: conflicting META records for step {s}")
        steps[s] = d''',
    )
    text = once(
        text,
        '        per_req = parse_fbw(fbw) if os.path.exists(fbw) else []\n        i = 0',
        '''        per_req = parse_fbw(fbw)
        with open(os.path.join(d, fn)) as driver:
            request_count = sum(bool(WL.match(line.strip())) for line in driver)
        _require_serial_trace_alignment(per_req, request_count, cell)
        i = 0''',
    )
    text = once(
        text,
        '            http = 400 if "HTTP400" in rest else 200',
        '            http, corrupted, outcome = _parse_step7_outcome(rest)',
    )
    text = once(text, '            nonfin = "nonfinite_lp=True" in rest\n', "")
    text = once(
        text,
        '''                "corrupted": int(bool(http == 400 or nonfin
                                      or (corr and int(corr.group(1)) > 0))),''',
        '''                "corrupted": int(corrupted),
                "request_outcome": outcome,''',
    )
    text = once(
        text,
        '        f = fb[i] if i < len(fb) else {}',
        '''        if i >= len(fb):
            raise ValueError(f"{jsonl}: no trace request {i} for build {b!r}; check the trace filename mapping")
        f = fb[i]''',
    )
    text = once(
        text,
        '            "corrupted": int(rec["corrupted"]),',
        '''            "corrupted": int(rec["corrupted"]),
            "request_outcome": _result_outcome(rec["http_status"], bool(rec["corrupted"])),''',
    )
    text = once(
        text,
        "    return rows\n\n\nCOLS =",
        '''    for build, count in seen.items():
        _require_serial_trace_alignment(fbw.get(build, []), count, f"{jsonl}/{build}")
    return rows


COLS =''',
    )
    text = once(
        text,
        '        "output_sha256"]',
        '        "output_sha256", "request_outcome"]',
    )
    text = once(
        text,
        'if __name__ == "__main__":',
        (ROOT / "payload/outcome_helpers.py").read_text()
        + '\n\nif __name__ == "__main__":',
    )
    ast.parse(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    target = args.repo / TARGET
    data = target.read_bytes()
    got = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if got != EXPECTED:
        parser.error(f"Source drift: expected {EXPECTED}, got {got}")
    old = data.decode()
    new = transform(old)
    compile(new, TARGET, "exec")
    diff = "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/" + TARGET,
            tofile="b/" + TARGET,
        )
    )
    if args.apply:
        target.write_text(new, encoding="utf-8")
        test_target = args.repo / "tests/test_outcomes.py"
        if test_target.exists():
            parser.error(f"{test_target} already exists; refusing to overwrite")
        test_target.write_text(
            (ROOT / "tests/test_outcomes.py").read_text(), encoding="utf-8"
        )
    print(diff, end="")


if __name__ == "__main__":
    main()
