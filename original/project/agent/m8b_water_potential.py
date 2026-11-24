"""Experimental bounded shaping from actor-visible facts only.

This module is not connected to a trainer or deployed policy. Parameters must
be selected and registered separately before any formal trial. It returns no
actions and never reads an environment, hidden map, model, or trajectory file.
"""
from __future__ import annotations

import math


LANDMARKS = "地标[别名,种类,dx,dy,路程,可用,再生步]"


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def water_potential(
    facts: dict,
    *,
    dead: bool,
    weight: float,
    urgency_threshold: float,
    distance_scale: float,
) -> float:
    """Return weight * (1 - missing-water fraction * bounded known distance).

    Unknown/unreachable water uses the distance cap. A nonnegative living
    potential avoids rewarding the final loss of a negative potential on
    death. At water, drinking does not remove the added potential; above the
    urgency threshold, it has no directional preference. The scalar is not a
    per-action bonus: only its duration-discounted difference may be used.
    """
    if (type(dead) is not bool or not isinstance(facts, dict)
            or not _finite(weight) or weight < 0
            or not _finite(urgency_threshold) or not 0 < urgency_threshold <= 100
            or not _finite(distance_scale) or distance_scale <= 0):
        raise ValueError("Invalid water-potential configuration or terminal flag")
    if dead:
        return 0.
    try:
        thirst = facts["剩余数值"]["水分"]
        landmarks = facts[LANDMARKS]
    except (KeyError, TypeError) as error:
        raise ValueError("Complete serialized actor water facts required") from error
    if not _finite(thirst) or not 0 <= thirst <= 100 or not isinstance(landmarks, list):
        raise ValueError("Invalid water reserve or observed-landmark list")
    distance = float(distance_scale)
    for landmark in landmarks:
        if not isinstance(landmark, list) or len(landmark) < 5:
            raise ValueError("Malformed observed landmark")
        if landmark[1] != "water":
            continue
        path_length = landmark[4]
        if not _finite(path_length) or (path_length < 0 and path_length != -1):
            raise ValueError("Invalid observed water path length")
        if path_length >= 0:
            distance = min(distance, path_length)
    urgency = max(0., 1. - thirst / urgency_threshold)
    return float(weight * (1. - urgency * (distance / distance_scale)))


def shaping_reward(
    before: float, after: float, elapsed_ticks: int, *, gamma_tick: float
) -> float:
    """Duration-aware potential difference; callers set dead-state Phi to 0.

    Across a complete naturally terminated trajectory, its discounted sum is
    -Phi(initial). Rollout boundaries must retain their real bootstrap and
    must not be treated as natural death.
    """
    if (not _finite(before) or before < 0 or not _finite(after) or after < 0
            or type(elapsed_ticks) is not int or elapsed_ticks < 1
            or not _finite(gamma_tick) or not 0 < gamma_tick <= 1):
        raise ValueError("Invalid potentials, duration, or per-tick discount")
    return gamma_tick**elapsed_ticks * after - before
