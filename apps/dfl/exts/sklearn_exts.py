import asyncio
from typing import Callable, Any

import numpy as np


def pairwise_custom_metric(custom_metric_func: Callable[[Any, Any], Any], us: list[Any], vs: list[Any] = None, reflexive: bool = False, symmetric: bool = False) -> np.ndarray:
	if vs is None:
		vs = us
	else:
		reflexive = False
		symmetric = False
	
	assert len(us) == len(vs)  # TODO: allow different-lengthen us and vs.
	
	ms = np.zeros((len(us), len(vs)), dtype=object)
	
	# Loop on the upper triangle (excluding the diagonal).
	upper_is, upper_js = np.triu_indices_from(ms, k=1)
	for i, j in zip(upper_is, upper_js):
		u, v = us[i], vs[j]
		ms[i][j] = custom_metric_func(u, v)
	
	# Loop on the diagonal.
	if reflexive:
		np.fill_diagonal(ms, 1.0)
	else:
		for i in range(len(us)):
			u, v = us[i], vs[i]
			ms[i][i] = custom_metric_func(u, v)
	
	# Loop on the lower triangle (excluding the diagonal).
	if symmetric:
		ms[upper_js, upper_is] = ms[upper_is, upper_js]
	else:
		lower_js, lower_is = upper_is, upper_js
		for i, j in zip(lower_is, lower_js):
			u, v = us[i], vs[j]
			ms[i, j] = custom_metric_func(u, v)
	
	return ms


async def async_concurrent_pairwise_custom_metric(custom_metric_func: Callable[[Any, Any], Any], us: list[Any], vs: list[Any] = None, reflexive: bool = False, symmetric: bool = False) -> np.ndarray:
	if vs is None:
		vs = us
	else:
		reflexive = False
		symmetric = False
	
	assert len(us) == len(vs)  # TODO: allow different-lengthen us and vs.
	
	ms = np.zeros((len(us), len(vs)), dtype=object)
	
	# noinspection PyShadowingNames
	async def async_custom_metric_func(u, v) -> Any:
		return await asyncio.to_thread(custom_metric_func, u, v)
	
	async with asyncio.TaskGroup() as tg:
		# Loop on the upper triangle (excluding the diagonal).
		upper_is, upper_js = np.triu_indices_from(ms, k=1)
		for i, j in zip(upper_is, upper_js):
			u, v = us[i], vs[j]
			ms[i][j] = tg.create_task(async_custom_metric_func(u, v))
		
		# Loop on the diagonal.
		if reflexive:
			np.fill_diagonal(ms, 1.0)
		else:
			for i in range(len(us)):
				u, v = us[i], vs[i]
				ms[i][i] = tg.create_task(async_custom_metric_func(u, v))
		
		# Loop on the lower triangle (excluding the diagonal).
		if symmetric:
			ms[upper_js, upper_is] = ms[upper_is, upper_js]
		else:
			lower_js, lower_is = upper_is, upper_js
			for i, j in zip(lower_is, lower_js):
				u, v = us[i], vs[j]
				ms[i, j] = tg.create_task(async_custom_metric_func(u, v))
	
	for idx, task in np.ndenumerate(ms):
		if isinstance(task, asyncio.Task):
			ms[idx] = await task
	
	return ms
