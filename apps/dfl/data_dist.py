import random
from random import Random

import numpy as np
import torch
from dfl.data import stratified_split


# Referred to as "IID" in McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
def iid(len_dataset: int, n: int, rng: Random = random) -> list[list[int]]:
	indices = list(range(len_dataset))
	rng.shuffle(indices)
	
	splits = [split.tolist() for split in np.array_split(indices, n)]
	return splits


def stratified(labels: list[int], n: int, rng: Random = random) -> list[tuple[int]]:
	# noinspection PyUnresolvedReferences
	splits = stratified_split(
		list(range(len(labels))),
		labels=labels,
		n_splits=n,
		rnd=rng
	)
	
	return list(splits)


def by_labels(labels: list[int], num_labels: int | None = None) -> list[list[int]]:
	if num_labels is None:
		num_labels = len(np.unique_values(labels))
	
	# noinspection PyUnresolvedReferences
	splits = [torch.nonzero(torch.as_tensor(labels) == i, as_tuple=False).squeeze().tolist() for i in range(num_labels)]
	return splits


# Referred to as "Non-IID" in McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
def non_iid(labels: list[int], n: int, shards_factor: int) -> list[list[int]]:
	sorted_indices = sorted(range(len(labels)), key=lambda idx: labels[idx])
	
	shards_count = shards_factor * n
	shards = [split.tolist() for split in np.array_split(sorted_indices, shards_count)]
	
	splits = [[] for _ in range(n)]
	for ni in range(n):
		for si in range(shards_factor):
			splits[ni].extend(shards[ni + (si * n)])
	
	return splits


def pretty_non_iid(*labelss: list[int], n: int, groups_count: int, rng: Random = random, unique_labels: list[int] | None = None) -> list[list[int]] | tuple[list[list[int]]]:
	if unique_labels is None:
		unique_labels = np.unique_values(np.concatenate(labelss)).tolist()
	else:
		unique_labels = unique_labels.copy()
	
	rng.shuffle(unique_labels)
	
	assert len(unique_labels) >= groups_count
	
	groups = [sorted(split.tolist()) for split in np.array_split(range(n), groups_count)]
	labels_per_groups = [split.tolist() for split in np.array_split(unique_labels, groups_count)]
	
	splits_by_labelss = tuple(by_labels(labels=labels, num_labels=len(unique_labels)) for labels in labelss)
	for splits_by_labels in splits_by_labelss:
		for indices in splits_by_labels:
			rng.shuffle(indices)
	
	splitss = tuple([[] for _ in range(n)] for _ in range(len(labelss)))
	for group_idx, (group, group_labels) in enumerate(zip(groups, labels_per_groups)):
		for label in group_labels:
			for labels_idx, splits_by_labels in enumerate(splits_by_labelss):
				label_splits = [split.tolist() for split in np.array_split(splits_by_labels[label], len(group))]
				
				for idx, n_idx in enumerate(group):
					splitss[labels_idx][n_idx].extend(label_splits[idx])
	
	if len(splitss) == 1:
		return splitss[0]
	else:
		# noinspection PyTypeChecker
		return splitss


def non_iid_strict_label_dir(labels: list[int], n: int, alpha: float = 0.5, np_rng: np.random.Generator = np.random, labels_proportions: dict[int, np.ndarray] | None = None, return_labels_proportions: bool = False) -> list[list[int]] | tuple[list[list[int]], dict[int, list[float]]]:
	labels = np.asarray(labels)
	unique_labels = np.unique(labels)
	
	labels_indices = {label: np.where(labels == label)[0] for label in unique_labels}
	for indices in labels_indices.values():
		np_rng.shuffle(indices)
	
	if labels_proportions is None:
		labels_proportions = {label: np_rng.dirichlet(np.repeat(alpha, n)) for label in unique_labels}
	
	indices = [[] for _ in range(n)]
	
	for label in unique_labels:
		label_indices = labels_indices[label]
		label_proportions = labels_proportions[label]
		
		counts = np.floor(label_proportions * len(label_indices)).astype(int)
		remaining = len(label_indices) - counts.sum()
		fractional = label_proportions * len(label_indices) - counts
		
		if remaining > 0:
			sorted_nodes = np.argsort(-fractional)[:remaining]
			counts[sorted_nodes] += 1
		
		splits_points = np.cumsum(counts)[:-1]
		splits = np.split(label_indices, splits_points)
		
		for i in range(n):
			indices[i].extend(splits[i].tolist())
	
	if not return_labels_proportions:
		return indices
	else:
		return indices, labels_proportions


def non_iid_label_dir(labels: list[int], n: int, alpha: float = 0.5, np_rng: np.random.Generator = np.random, labels_proportions: dict[int, np.ndarray] | None = None, return_labels_proportions: bool = False, min_required_size: int = 10) -> list[list[int]] | tuple[list[list[int]], dict[int, list[float]]]:
	labels = np.asarray(labels)
	unique_labels = np.unique(labels)
	
	labels_indices = {label: np.where(labels == label)[0] for label in unique_labels}
	
	for indices in labels_indices.values():
		np_rng.shuffle(indices)
	
	nodes_indices = None
	if labels_proportions is not None:
		nodes_indices = [[] for _ in range(n)]
		
		for label in unique_labels:
			label_indices = labels_indices[label]
			label_proportions = labels_proportions[label]
			
			splits_points = (np.cumsum(label_proportions) * len(label_indices)).astype(int)[:-1]
			splits = np.split(label_indices, splits_points)
			
			for i in range(n):
				nodes_indices[i].extend(splits[i].tolist())
	else:
		min_size = 0
		while min_size < min_required_size:
			nodes_indices = [[] for _ in range(n)]
			labels_proportions = {}
			
			for label in unique_labels:
				label_indices = labels_indices[label]
				
				label_proportions = np_rng.dirichlet(np.repeat(alpha, n))
				
				# Balance.
				mask = np.array([len(indices) < len(labels) / n for indices in nodes_indices])
				label_proportions = label_proportions * mask
				label_proportions /= label_proportions.sum()
				
				labels_proportions[label] = label_proportions
				
				splits_points = (np.cumsum(label_proportions) * len(label_indices)).astype(int)[:-1]
				splits = np.split(label_indices, splits_points)
				
				for i in range(n):
					nodes_indices[i].extend(splits[i].tolist())
				
				min_size = min([len(idx) for idx in nodes_indices])
	
	if not return_labels_proportions:
		return nodes_indices
	else:
		return nodes_indices, labels_proportions


def pathological_non_iid(*labelss: list[int], n: int, k: int, np_rng: np.random.Generator = np.random, unique_labels: list[int] | None = None) -> list[list[int]] | tuple[list[list[int]]]:
	assert k >= 1
	
	labelss = [np.asarray(labels) for labels in labelss]
	
	if unique_labels is None:
		unique_labels = np.unique_values(np.concatenate(labelss))
	else:
		unique_labels = np.asarray(unique_labels)
	
	assert k * n >= len(unique_labels)
	
	# Assert `unique_labels` are 0, 1, 2, ....
	
	nodes_labels = [set() for _ in range(n)]
	for node in range(n):
		node_labels = nodes_labels[node]
		
		node_labels.add(node % len(unique_labels))  # This ensures that no label is left out in the returned data distribution.
		
		for i in range(k - 1):
			while True:
				label = np_rng.integers(0, len(unique_labels))
				if label not in node_labels:
					node_labels.add(label)
					break
	
	labels_nodes = [set() for _ in range(len(unique_labels))]
	for node, node_labels in enumerate(nodes_labels):
		for label in node_labels:
			labels_nodes[label].add(node)
	
	labelss_indices = [[np.where(labels == i)[0] for i in range(len(unique_labels))] for labels in labelss]
	for labels_indices in labelss_indices:
		for label_indices in labels_indices:
			np_rng.shuffle(label_indices)
	
	nodes_indicess = tuple([[] for _ in range(n)] for _ in range(len(labelss)))
	for label in unique_labels:
		for (r, labels_indices) in enumerate(labelss_indices):
			indices = labels_indices[label]
			
			nodes = labels_nodes[label]
			splits = np.array_split(indices, len(nodes))
			
			for (node, split) in zip(nodes, splits):
				nodes_indicess[r][node].extend(split.tolist())
	
	if len(nodes_indicess) == 1:
		return nodes_indicess[0]
	else:
		# noinspection PyTypeChecker
		return nodes_indicess
