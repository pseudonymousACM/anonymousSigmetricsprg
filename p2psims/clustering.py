from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity


async def affinity_propagation(data: dict[str, Iterable[float] | torch.Tensor | np.ndarray]) -> dict[str, set[str]]:
	# Assume that data is an ordered dictionary.
	
	data_values = list(data.values())
	
	ap = AffinityPropagation()
	ap.fit(data_values)
	
	clusters = defaultdict(set)
	for data_key, cluster_index in zip(data.keys(), ap.labels_):
		cluster_index = int(cluster_index)
		
		clusters[cluster_index].add(data_key)
	
	clusters = dict(clusters)
	
	return clusters


async def affinity_propagation_cosine(data: dict[str, Iterable[float] | torch.Tensor | np.ndarray]) -> dict[str, set[str]]:
	# Assume that data is an ordered dictionary.
	
	data_values = list(data.values())
	data_values = cosine_similarity(data_values)
	
	ap = AffinityPropagation(affinity='precomputed')
	ap.fit(data_values)
	
	clusters = defaultdict(set)
	for data_key, cluster_index in zip(data.keys(), ap.labels_):
		cluster_index = int(cluster_index)
		
		clusters[cluster_index].add(data_key)
	
	clusters = dict(clusters)
	
	return clusters
