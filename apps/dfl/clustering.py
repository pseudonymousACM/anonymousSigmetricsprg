import asyncio
import functools
from collections import defaultdict
from typing import Callable

import torch
from kmedoids import KMedoids
from sklearn.cluster import AffinityPropagation, KMeans

import conf
import similarity
from exts.sklearn_exts import async_concurrent_pairwise_custom_metric


async def affinity_propagation(data: dict[str, torch.Tensor], affinity_custom_metric_func: Callable[[torch.Tensor, torch.Tensor], float], reflexive: bool = False, symmetric: bool = False) -> dict[str, set[str]]:
	# Assume that data is an ordered dictionary.
	
	data_values = list(data.values())
	data_values = await async_concurrent_pairwise_custom_metric(affinity_custom_metric_func, data_values, reflexive=reflexive, symmetric=symmetric)
	
	ap = AffinityPropagation(affinity='precomputed', copy=False)
	await asyncio.to_thread(ap.fit, data_values)
	
	clusters = defaultdict(set)
	for data_key, cluster_index in zip(data.keys(), ap.labels_):
		cluster_index = int(cluster_index)
		
		clusters[cluster_index].add(data_key)
	
	clusters = dict(clusters)
	
	return clusters


async def affinity_propagation_cosine(data: dict[str, torch.Tensor]) -> dict[str, set[str]]:
	return await affinity_propagation(data, similarity.cosine, reflexive=True, symmetric=True)


async def affinity_propagation_euclidean(data: dict[str, torch.Tensor]) -> dict[str, set[str]]:
	return await affinity_propagation(data, similarity.neg_sq_euclid, reflexive=True, symmetric=True)


async def kmeans(data: dict[str, torch.Tensor], k: int) -> dict[str, set[str]]:
	# Assume that data is an ordered dictionary.
	
	data_values = list(data.values())
	data_values = await asyncio.to_thread(lambda: [v.to_dense() for v in data_values])  # TODO: KMeans does not support sparse tensors.
	
	km = KMeans(n_clusters=min(k, len(data)))  # TODO: a data copy also happens in here (`copy_x=True`).
	labels = await asyncio.to_thread(km.fit_predict, data_values)
	
	clusters = defaultdict(set)
	for data_key, cluster_index in zip(data.keys(), labels):
		cluster_index = int(cluster_index)
		
		clusters[cluster_index].add(data_key)
	
	clusters = dict(clusters)
	
	return clusters


async def kmedoids(data: dict[str, torch.Tensor], k: int, similarity_custom_metric_func: Callable[[torch.Tensor, torch.Tensor], float], reflexive: bool = False, symmetric: bool = False) -> dict[str, set[str]]:
	# Assume that data is an ordered dictionary.
	
	data_values = list(data.values())
	data_values = await async_concurrent_pairwise_custom_metric(similarity_custom_metric_func, data_values, reflexive=reflexive, symmetric=symmetric)
	# noinspection PyTypeChecker
	data_values = await asyncio.to_thread(data_values.astype, float, copy=False)  # TODO: this does a copy anyway due to the required casting from object.
	
	km = KMedoids(n_clusters=min(k, len(data)), method='pam')  # FIXME: should use `fasterpam`, but it has a bug not returning all clusters, e.g. when k equals the number of data.
	labels = await asyncio.to_thread(km.fit_predict, data_values)
	
	clusters = defaultdict(set)
	for data_key, cluster_index in zip(data.keys(), labels):
		cluster_index = int(cluster_index)
		
		clusters[cluster_index].add(data_key)
	
	clusters = dict(clusters)
	
	return clusters


async def cosine_kmedoids(data: dict[str, torch.Tensor], k: int) -> dict[str, set[str]]:
	return await kmedoids(data, k, similarity.cosine_dist, reflexive=True, symmetric=True)


async def euclid_kmedoids(data: dict[str, torch.Tensor], k: int) -> dict[str, set[str]]:
	return await kmedoids(data, k, similarity.euclid_dist, reflexive=True, symmetric=True)


match conf.CLUSTERING:
	case 'affinity-propagation':
		match conf.SIMILARITY:
			case 'cosine' | 'cosine-raw':
				_f = affinity_propagation_cosine
			case 'euclidean':
				_f = affinity_propagation_euclidean
			# FIXME: missing the `cosine-raw` case. Maybe just directly use `similarity.calc` (do not forget to update the corresponding documentations in the readme)?
			case _:
				assert False
	case 'affinity-propagation-euclidean':
		_f = affinity_propagation_euclidean
	case 'affinity-propagation-cosine':
		_f = affinity_propagation_cosine
	case _:
		if conf.CLUSTERING.startswith('kmeans-'):
			_k = int(conf.CLUSTERING.split('-', 1)[1])
			_f = functools.partial(kmeans, k=_k)
		elif conf.CLUSTERING.startswith('kmedoids-'):
			_splits = conf.CLUSTERING.split('-', 2)
			
			_k = int(_splits[1])
			_metric = None if len(_splits) == 2 else _splits[2]
			
			match _metric:
				case None:
					match conf.SIMILARITY:
						case 'cosine':
							_f = cosine_kmedoids
						case 'euclidean':
							_f = euclid_kmedoids
						# FIXME: missing the `cosine-raw` case. Maybe just directly use `similarity.calc` (do not forget to update the corresponding documentations in the readme)?
						case _:
							assert False
				case 'cosine':
					_f = cosine_kmedoids
				case 'euclid':
					_f = euclid_kmedoids
				case _:
					assert False
			
			_f = functools.partial(_f, k=_k)
		else:
			assert False


async def do(data: dict[str, torch.Tensor]) -> dict[str, set[str]]:
	return await _f(data)
