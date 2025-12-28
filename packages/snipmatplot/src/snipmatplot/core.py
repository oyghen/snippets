__all__ = ["discrete_cmap"]

from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np

RGBA: TypeAlias = tuple[float, float, float, float]


def discrete_cmap(
    n: int,
    /,
    *,
    name: str = "viridis_r",
    lower_bound: float = 0.05,
    upper_bound: float = 0.9,
) -> list[RGBA]:
    """Create discrete colormap values.

    Examples
    --------
    >>> import numpy as np
    >>> import snipmatplot
    >>> np.set_printoptions(legacy="1.21")
    >>> snipmatplot.core.discrete_cmap(2)
    [(0.876168, 0.891125, 0.09525, 1.0), (0.282623, 0.140926, 0.457517, 1.0)]
    """
    cmap = plt.get_cmap(name)
    return [cmap(i) for i in np.linspace(lower_bound, upper_bound, num=n)]
