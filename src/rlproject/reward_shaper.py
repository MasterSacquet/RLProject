"""Safety-aware reward shaping utilities for Highway-v0 training.

This module implements the extension formula:

    shaped_reward = baseline_reward - lambda * collision_penalty - mu * crash_indicator

with two predefined variants:
    - conservative: lambda=10, mu=50
    - moderate: lambda=5, mu=30
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SafetyRewardWeights:
    """Weights used in the safety-aware shaping penalty."""

    lambda_collision: float
    mu_crash: float


CONSERVATIVE_WEIGHTS = SafetyRewardWeights(lambda_collision=10.0, mu_crash=50.0)
MODERATE_WEIGHTS = SafetyRewardWeights(lambda_collision=5.0, mu_crash=30.0)

PRESET_WEIGHTS = {
    "conservative": CONSERVATIVE_WEIGHTS,
    "moderate": MODERATE_WEIGHTS,
}


def get_safety_weights(variant: str) -> SafetyRewardWeights:
    """Return one of the predefined shaping presets.

    Args:
        variant: Either "conservative" or "moderate".

    Raises:
        ValueError: If an unknown variant is requested.
    """

    key = variant.strip().lower()
    if key not in PRESET_WEIGHTS:
        supported = ", ".join(sorted(PRESET_WEIGHTS.keys()))
        raise ValueError(f"Unknown safety variant '{variant}'. Supported: {supported}")
    return PRESET_WEIGHTS[key]


def extract_crash_indicator(
    info: Optional[Mapping[str, Any]] = None,
    crashed: Optional[bool] = None,
) -> float:
    """Compute a binary crash indicator from env outputs.

    The function is defensive because different env wrappers expose different
    keys in ``info``. If ``crashed`` is provided explicitly, it has priority.
    """

    if crashed is not None:
        return 1.0 if crashed else 0.0

    if info is None:
        return 0.0

    for key in ("crashed", "collision", "is_collision", "collision_occurred"):
        value = info.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0

    return 0.0


def shape_safety_reward(
    baseline_reward: float,
    *,
    weights: SafetyRewardWeights,
    collision_penalty: float,
    crash_indicator: float,
) -> float:
    """Apply the safety-aware reward equation.

    Args:
        baseline_reward: Original reward emitted by Highway-v0.
        weights: Safety reward coefficients (lambda, mu).
        collision_penalty: Positive collision cost magnitude.
        crash_indicator: 1.0 if a crash occurred, else 0.0.
    """

    penalty = weights.lambda_collision * collision_penalty + weights.mu_crash * crash_indicator
    return float(baseline_reward - penalty)


def apply_safety_shaping(
    baseline_reward: float,
    info: Optional[Mapping[str, Any]] = None,
    *,
    variant: str = "moderate",
    collision_penalty: float = 1.5,
    crashed: Optional[bool] = None,
) -> float:
    """High-level helper to shape one reward step.

    Notes:
        ``collision_penalty`` should be provided as a positive magnitude.
        In the current shared config, ``collision_reward`` is -1.5, so the
        corresponding magnitude is 1.5.
    """

    weights = get_safety_weights(variant)
    crash_indicator = extract_crash_indicator(info=info, crashed=crashed)

    # on applique la collision magnitude seulement si on détecte un crash
    effective_collision_penalty = crash_indicator * float(collision_penalty)

    return shape_safety_reward(
        baseline_reward,
        weights=weights,
        collision_penalty=effective_collision_penalty,
        crash_indicator=crash_indicator,
    )
