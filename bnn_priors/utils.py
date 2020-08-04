import math
from typing import Callable


def get_cosine_schedule(samples_per_cycle: int) -> Callable[[int], float]:
    def schedule(i: int) -> float:
        cycle_progress = (i % samples_per_cycle) / samples_per_cycle
        scale = 0.5 * (math.cos(math.pi * cycle_progress) + 1.)
        return scale
    return schedule
