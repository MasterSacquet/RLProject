"""Safety evaluation metrics for Highway-v0 agents.

Each evaluation function takes a list of episode records. An episode record
is a dict with at least the following keys:

    {
        "reward": float,
        "crashed": bool,
        "min_distance": float or None,
    }

These records are produced by run_episode() defined in this module, which
follows the same structure as evaluate_custom() in evaluate_multiseed.py.
"""

from typing import Any, Mapping, Optional
import numpy as np


def collision_rate(episodes: list[dict]) -> float:
    """Fraction of episodes in which at least one collision occurred.

    Args:
        episodes: List of episode records.

    Returns:
        Value in [0, 1].
    """
    crashed_count = sum(1 for ep in episodes if ep["crashed"])
    return crashed_count / len(episodes)


def mean_crashes(episodes: list[dict]) -> float:
    """Mean number of crashes per episode.

    In highway-v0 an episode ends on the first crash, so this is equivalent
    to collision_rate() when each record stores a single boolean. It is
    included as a separate metric for compatibility with wrappers that may
    count multiple collision events per episode.

    Args:
        episodes: List of episode records.

    Returns:
        Mean crash count across episodes.
    """
    return float(np.mean([ep["crashed"] for ep in episodes]))


def mean_safety_margin(episodes: list[dict]) -> Optional[float]:
    """Mean of the per-episode minimum distance to any obstacle.

    Returns None if no episode provided a distance measurement.

    Args:
        episodes: List of episode records.

    Returns:
        Mean minimum distance, or None.
    """
    distances = [ep["min_distance"] for ep in episodes if ep["min_distance"] is not None]
    if not distances:
        return None
    return float(np.mean(distances))


def reward_stats(episodes: list[dict]) -> dict:
    """Descriptive statistics for per-episode total rewards.

    Args:
        episodes: List of episode records.

    Returns:
        Dict with keys mean, std, min, max, median.
    """
    rewards = [ep["reward"] for ep in episodes]
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
    }


def compute_safety_summary(episodes: list[dict]) -> dict:
    """Aggregate all safety and performance metrics for a set of episodes.

    Args:
        episodes: List of episode records.

    Returns:
        Dict with keys: collision_rate, mean_crashes, safety_margin,
        reward_mean, reward_std, reward_min, reward_max, reward_median.
    """
    stats = reward_stats(episodes)

    speeds = [ep["mean_speed"] for ep in episodes if ep["mean_speed"] is not None]
    mean_speed = float(np.mean(speeds)) if speeds else None

    return {
        "collision_rate": collision_rate(episodes),
        "mean_crashes": mean_crashes(episodes),
        "safety_margin": mean_safety_margin(episodes),
        "mean_speed": mean_speed,
        "reward_mean": stats["mean"],
        "reward_std": stats["std"],
        "reward_min": stats["min"],
        "reward_max": stats["max"],
        "reward_median": stats["median"],
    }


def _extract_min_distance(info: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Extract a proximity measure from the env info dict.

    Highway-v0 does not expose a direct obstacle distance in info, so this
    function checks several plausible keys and returns None when none is found.
    The calling code should fall back to None gracefully.

    Args:
        info: Info dict returned by env.step().

    Returns:
        Minimum distance as a float, or None.
    """
    if info is None:
        return None
    for key in ("min_distance", "distance_to_obstacle", "ttc"):
        value = info.get(key)
        if value is not None:
            return float(value)
    return None


def run_episode(env, select_action, seed=None) -> dict:
    """Run one episode and return a safety-annotated record.

    This function mirrors the evaluation loop in evaluate_multiseed.py and
    is intended to be used in evaluation scripts for safety-aware agents.

    Args:
        env: A gymnasium environment (already reset externally is fine, but
             this function calls reset() itself with no seed).
        select_action: Callable that takes an observation array and returns
                       an integer action.

    Returns:
        Episode record dict with keys: reward, crashed, min_distance, length.
    """
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    steps = 0
    crashed = False
    min_distance = None
    speeds = []
    done = False
    truncated = False

    while not (done or truncated):
        action = select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += float(reward)
        steps += 1

        if info.get("crashed", False):
            crashed = True

        dist = _extract_min_distance(info)
        if dist is not None:
            min_distance = dist if min_distance is None else min(min_distance, dist)

        speed = info.get("speed")
        if speed is not None:
            speeds.append(float(speed))

    return {
        "reward": episode_reward,
        "crashed": crashed,
        "min_distance": min_distance,
        "length": steps,
        "mean_speed":   float(np.mean(speeds)) if speeds else None,
    }