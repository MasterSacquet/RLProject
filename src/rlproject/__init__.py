"""Core RLProject package: custom DQN agent, replay buffer, and environment wrapper."""

from .reward_shaper import (
	CONSERVATIVE_WEIGHTS,
	MODERATE_WEIGHTS,
	PRESET_WEIGHTS,
	SafetyRewardWeights,
	apply_safety_shaping,
	extract_crash_indicator,
	get_safety_weights,
	shape_safety_reward,
)

__all__ = [
	"SafetyRewardWeights",
	"CONSERVATIVE_WEIGHTS",
	"MODERATE_WEIGHTS",
	"PRESET_WEIGHTS",
	"get_safety_weights",
	"extract_crash_indicator",
	"shape_safety_reward",
	"apply_safety_shaping",
]
