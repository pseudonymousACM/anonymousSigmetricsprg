import asyncio
import functools

import torch

import conf
from exts.torch_exts import mean_sparse_tensors, mean_weighted_sparse_tensors


def mean(ts: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, float]:
	return mean_sparse_tensors([t for (t, _) in ts]), sum(w for (_, w) in ts)


mean_weighted = mean_weighted_sparse_tensors


async def mean_weighted_pre_norm(ts: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, float]:
	# TODO: compute the norms concurrently.
	ts = [(t, w / await asyncio.to_thread(torch.norm, t, p=2)) for (t, w) in ts]
	return await asyncio.to_thread(mean_weighted_sparse_tensors, ts)


match conf.SON_AGG:
	case 'mean':
		do = functools.partial(asyncio.to_thread, mean)
	case 'mean-w':
		do = functools.partial(asyncio.to_thread, mean_weighted)
	case 'mean-w-pnorm':
		do = mean_weighted_pre_norm
	case _:
		assert False
