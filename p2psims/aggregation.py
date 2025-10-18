from typing import Iterable

import numpy as np
import torch


async def np_mean(data: Iterable[float | Iterable[float] | np.ndarray]) -> np.ndarray:
	data = [np.asanyarray(datum) for datum in data]
	return np.mean(np.stack(data), axis=0)


async def torch_mean(data: Iterable[float | Iterable[float] | torch.Tensor]) -> torch.Tensor:
	data = [torch.as_tensor(datum) for datum in data]
	return torch.mean(torch.stack(data), dim=0)
