import numpy as np


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
        self.rng = np.random.default_rng(seed=2023)

    def __call__(self):
        return self.rng.choice(self.values)