import functools
import math
import sys
from random import Random
from typing import Callable, Any

import networkx as nx


def _ensure_connected(graph_factory: Callable[[Any], nx.Graph], start_seed: int = 42, max_tries: int = 10_000) -> nx.Graph:
	rnd = Random(start_seed)
	
	g = None
	for _ in range(max_tries):
		seed = rnd.randint(-sys.maxsize, sys.maxsize)
		
		# noinspection PyArgumentList
		g = graph_factory(seed=seed)
		if nx.is_connected(g):
			break
		else:
			g = None
	
	if g is None:
		raise ValueError("Max tries reached.")
	else:
		return g


def rgg(n: int = 10, start_seed=42) -> nx.Graph:
	r = math.sqrt((math.log(n) + 1) / (math.pi * n))
	g_factory = functools.partial(nx.random_geometric_graph, n=n, radius=r)
	g = _ensure_connected(g_factory, start_seed=start_seed)
	return g


def erg(n: int = 10, start_seed=42) -> nx.Graph:
	p = math.log(n) / n + 0.05
	g_factory = functools.partial(nx.erdos_renyi_graph, n=n, p=p)
	g = _ensure_connected(g_factory, start_seed=start_seed)
	
	return g


def bag(n: int = 10, m: int = 1, seed=42) -> nx.Graph:
	return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def wsg(n: int = 10, k: int = 2, p: float = 0.2, start_seed=42) -> nx.Graph:
	g_factory = functools.partial(nx.watts_strogatz_graph, n=n, k=k, p=p)
	return _ensure_connected(g_factory, start_seed)


def rst(n: int = 10, seed=42) -> nx.Graph:
	cg = nx.complete_graph(n)
	return nx.random_spanning_tree(cg, seed=seed)
