"""SUP-008 opt-in participation audit. Pure Python: safe to unit-test without Flower.

This module checks *logical* partition identity, not proxy IDs, and never performs
aggregation. When the opt-in flag is absent, the historical Flower code never
calls these functions.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def validate_sup008_round(
    *,
    server_round: int,
    expected_sizes: Mapping[int, int],
    planned_by_proxy: Mapping[str, int],
    results: Sequence[Mapping[str, object]],
    failures_count: int,
) -> dict:
    expected_ids = set(range(5))
    if not (1 <= int(server_round) <= 50):
        raise RuntimeError("SUP008: invalid server round")
    if set(expected_sizes) != expected_ids:
        raise RuntimeError("SUP008: expected exactly five logical partitions")
    if any(int(n) <= 0 for n in expected_sizes.values()):
        raise RuntimeError("SUP008: non-positive partition size")
    if sum(int(n) for n in expected_sizes.values()) != 12000:
        raise RuntimeError("SUP008: incorrect canonical training total")
    if len(planned_by_proxy) != 5 or set(planned_by_proxy.values()) != expected_ids:
        raise RuntimeError("SUP008: planned participants are incomplete/duplicated")
    if failures_count != 0 or len(results) != 5:
        raise RuntimeError("SUP008: missing or failed client fit")

    actual_by_proxy = {}
    actual_sizes = {}
    for fit in results:
        cid = str(fit['cid'])
        try:
            logical = int(fit['logical_client_id'])
            samples = int(fit['num_examples'])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("SUP008: invalid fit identity/sample count") from exc
        if cid in actual_by_proxy or logical in actual_sizes:
            raise RuntimeError("SUP008: duplicate proxy or logical partition")
        if logical not in expected_ids or planned_by_proxy.get(cid) != logical:
            raise RuntimeError("SUP008: unexpected proxy-to-partition identity")
        if samples != int(expected_sizes[logical]) or samples <= 0:
            raise RuntimeError("SUP008: fit sample count mismatch")
        actual_by_proxy[cid] = logical
        actual_sizes[logical] = samples

    if set(actual_by_proxy) != set(planned_by_proxy) or set(actual_sizes) != expected_ids:
        raise RuntimeError("SUP008: incomplete planned participants")
    total = sum(actual_sizes.values())
    if total != 12000:
        raise RuntimeError("SUP008: invalid summed fit sample count")
    weights = {str(k): v / total for k, v in actual_sizes.items()}
    if not all(math.isfinite(w) and w > 0 for w in weights.values()):
        raise RuntimeError("SUP008: non-positive/non-finite aggregation weight")
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise RuntimeError("SUP008: aggregation weights do not sum to one")

    return {
        'round': int(server_round),
        'planned_cids': sorted(planned_by_proxy),
        'actual_cids': sorted(actual_by_proxy),
        'planned_logical_client_ids': sorted(planned_by_proxy.values()),
        'actual_logical_client_ids': sorted(actual_sizes),
        'logical_client_by_proxy': dict(sorted(actual_by_proxy.items())),
        'samples_by_logical_client': {str(k): v for k, v in sorted(actual_sizes.items())},
        'aggregation_weight_by_logical_client': dict(sorted(weights.items())),
        'sample_total': total,
        'num_failures': 0,
    }
