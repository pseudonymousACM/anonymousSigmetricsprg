import asyncio
import functools
import math

import torch

import conf


def cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8, normalize: bool = False, return_tensor: bool = False) -> float | torch.Tensor:
	norm_u = torch.norm(u, p=2)
	norm_v = torch.norm(v, p=2)
	
	norm_u.clamp_min_(eps)
	norm_v.clamp_min_(eps)
	
	cosine_v = ((u / norm_u) * (v / norm_v)).sum()
	
	if normalize:
		res = (cosine_v + 1) / 2  # Min-max normalization from [-1, 1] to [0, 1].
	else:
		res = cosine_v
	
	torch.clip_(res, -1, 1)
	return res.item() if not return_tensor else res


def inv_euclid(u: torch.Tensor, v: torch.Tensor) -> float:
	euclid_dist = torch.norm(u - v, p=2)
	
	res = 1 / (1 + euclid_dist)
	
	torch.clip_(res, 0, 1)
	return res.item()


def neg_sq_euclid(u: torch.Tensor, v: torch.Tensor) -> float:
	res = -torch.sum((u - v) ** 2)
	
	torch.clip_(res, -math.inf, 0)
	return res.item()


# Not a similarity, but a distance metric.
def euclid_dist(u: torch.Tensor, v: torch.Tensor) -> float:
	res = torch.sum((u - v) ** 2)
	res.sqrt_()
	
	torch.clip_(res, 0, math.inf)
	return res.item()


# Not a similarity, but a distance metric.
def cosine_dist(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> float:
	res = 1 - cosine(u, v, eps, return_tensor=True)
	
	torch.clip_(res, 0, 2)
	return res.item()


match conf.SIMILARITY:
	case 'cosine':
		_f = functools.partial(cosine, normalize=True)
	case 'euclidean':
		_f = inv_euclid
	case 'cosine-raw':
		_f = cosine
	case _:
		assert False


async def acalc(u: torch.Tensor, v: torch.Tensor) -> float:
	# noinspection PyTypeChecker
	return await asyncio.to_thread(_f, u, v)


async def calc(u: torch.Tensor, v: torch.Tensor) -> float:
	return _f(u, v)
