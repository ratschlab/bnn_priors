import numpy as np


def get_cosine_schedule(samples_per_cycle):
    def schedule(i):
        cycle_progress = (i % samples_per_cycle) / samples_per_cycle
        scale = 0.5 * (np.cos(np.pi * cycle_progress) + 1.)
        return scale
    return schedule