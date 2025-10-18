import asyncio
from typing import Callable, Awaitable, Any, override

import numpy as np
import torch
from dfl import Aggregator, Meta
from dfl.aggregation.torch import StateDictAggregator, TensorToDeviceDecoratingAggregator, WeightedAvgTensorsAggregator

import clustering
import conf
import conf_eval
import search_data
import similarity
from exts.dfl_exts import DataClusteringAggregator, StateDictToDeviceDecoratingAggregator


async def _state_dicts_to_search_data(data: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
	res = {}
	async with asyncio.TaskGroup() as tg:
		for k, sd in data.items():
			res[k] = tg.create_task(search_data.build(sd, cpu=False))
	
	for k, task in res.items():
		res[k] = await task
	
	return res


# noinspection PyShadowingNames
def fedavg_agg_factory(**kwargs):
	return StateDictAggregator(lambda **kwargs: TensorToDeviceDecoratingAggregator(conf_eval.AGGREGATOR_DEVICE_EVAL, WeightedAvgTensorsAggregator(weight_meta_key='data_len')))


async def _cluster_state_dicts(data: dict[str, dict[str, torch.Tensor]], cluster: Callable[[dict[str, Any]], Awaitable[dict[str, set[str]]]] | None = None) -> dict[str, set[str]]:
	if cluster is None:
		cluster = clustering.do
	
	data = await _state_dicts_to_search_data(data)
	res = await cluster(data)
	return res


def clustering_agg_factory(**kwargs):
	return StateDictToDeviceDecoratingAggregator(
		device=conf_eval.AGGREGATOR_DEVICE_EVAL,
		agg=DataClusteringAggregator(
			intra_cluster_agg_factory=fedavg_agg_factory,
			cluster=_cluster_state_dicts,
			key='from'
		),
		inplace=True
	)


class IdealTopKAggregator(Aggregator):
	def __init__(self, k: int, intra_agg_factory: Callable[..., Aggregator], similarity_func: Callable[[Any, Any], Awaitable[float]], key: str):
		self._k = k
		self._intra_agg_factory = intra_agg_factory
		self._similarity_func = similarity_func
		self._key = key
		
		self._index = 0
		self._datas = list[Any]()
		self._metas = list[Meta]()
		self._sims = list[list[float]]()
		
		self._lock = asyncio.Lock()
	
	@override
	async def add(self, meta: Meta | None = None, data: Any = None) -> None:
		async with self._lock:
			my_index = self._index
			self._index += 1
			
			self._datas.append(data)
			self._metas.append(meta)
			my_sims = []
			self._sims.append(my_sims)
		
		async with asyncio.TaskGroup() as tg:
			for i in range(my_index):
				# noinspection PyTypeChecker
				my_sims.append(tg.create_task(self._similarity_func(data, self._datas[i])))
			
			for i in range(my_index):
				my_sims[i] = await my_sims[i]
	
	@override
	async def aggregate(self) -> dict[Any, tuple[set[Any], Any]]:
		ms = np.zeros((len(self._datas), len(self._datas)), dtype=float)
		np.fill_diagonal(ms, -np.inf)  # Avoids selecting self.
		
		upper_is, upper_js = np.triu_indices_from(ms, k=1)
		
		lower_is, lower_js = upper_js, upper_is
		for i, j in zip(lower_is, lower_js):
			ms[i][j] = self._sims[i][j]
		
		ms[upper_is, upper_js] = ms[lower_is, lower_js]
		
		topks = []
		for i in range(len(self._datas)):
			topks.append(np.argpartition(ms[i], -self._k)[-self._k:].tolist())
		
		res = dict()
		async with asyncio.TaskGroup() as tg:
			for idx, topk_indices in enumerate(topks):
				key = self._metas[idx][self._key]
				res[key] = tg.create_task(self._aggregate_one(idx, topk_indices))
			
			for key, task in res.items():
				res[key] = await task
		
		return res
	
	async def _aggregate_one(self, idx: int, topk_indices: list[int]) -> tuple[set[Any], Any]:
		agg = self._intra_agg_factory()
		meta = self._metas[idx]
		
		async with asyncio.TaskGroup() as tg:
			tg.create_task(agg.add(meta, self._datas[idx]))
			for top_idx in topk_indices:
				tg.create_task(agg.add(self._metas[top_idx], self._datas[top_idx]))
		
		topk_keys = {self._metas[idx][self._key] for idx in topk_indices}
		return topk_keys, await agg.aggregate()


class _StateDictToSearchDataDecoratingAggregator(Aggregator):
	def __init__(self, agg: Aggregator):
		self._agg = agg
	
	async def add(self, meta: Meta | None = None, data=None) -> None:
		data = await search_data.build(data, cpu=False)
		await self._agg.add(meta, data)
	
	async def aggregate(self) -> Any:
		return await self._agg.aggregate()


async def _similarity_state_dict(u, v, f=None) -> float:
	if f is None:
		f = similarity.acalc
	
	async with asyncio.TaskGroup() as tg:
		u = tg.create_task(search_data.build(u, cpu=False))
		v = tg.create_task(search_data.build(v, cpu=False))
		
		u = await u
		v = await v
	
	return await f(u, v)


def ideal_topk_agg_factory(k: int, **kwargs):
	return StateDictToDeviceDecoratingAggregator(
		device=conf_eval.AGGREGATOR_DEVICE_EVAL,
		agg=IdealTopKAggregator(
			k=k,
			intra_agg_factory=fedavg_agg_factory,
			similarity_func=_similarity_state_dict,
			key='from'
		),
		inplace=True
	)


match conf.CENTRALIZED_AGGREGATOR:
	case 'fedavg':
		agg_factory = fedavg_agg_factory
	case 'clustering':
		agg_factory = clustering_agg_factory
	case _:
		assert False
