from numbers import Number
from typing import Callable

import numpy as np


def mat_to_indices(mat: np.ndarray, criteria: Callable[[Number], bool], include_self: bool = False) -> list[np.ndarray]:
	indices = [
		np.where(np.vectorize(criteria)(mat[i]) & ((np.arange(mat.shape[0]) != i) if not include_self else True))[0]
		for i in range(mat.shape[0])
	]
	return indices
