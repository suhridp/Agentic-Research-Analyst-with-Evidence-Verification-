"""Small statistics helpers so every reported rate carries uncertainty."""
from __future__ import annotations
import math, random

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def bootstrap_mean(xs: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(iters))
    return (sum(xs) / len(xs), means[int(.025 * iters)], means[int(.975 * iters) - 1])
