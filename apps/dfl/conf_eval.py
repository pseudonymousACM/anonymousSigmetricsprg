import functools
from random import Random

import networkx as nx
import torch

import conf
import latsinit
import topogen

if conf.DEVICE is None:
	DEVICE_EVAL = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
	DEVICE_EVAL = torch.device(conf.DEVICE)

if conf.AGGREGATOR_DEVICE is None:
	AGGREGATOR_DEVICE_EVAL = DEVICE_EVAL
else:
	AGGREGATOR_DEVICE_EVAL = torch.device(conf.AGGREGATOR_DEVICE)

if conf.SON_DATA_DEVICE is None:
	# TODO: make the default value same as the `DEVICE`.
	SON_DATA_DEVICE_EVAL = torch.device('cpu')
else:
	SON_DATA_DEVICE_EVAL = torch.device(conf.SON_DATA_DEVICE)

match conf.INITIAL_PEERS:
	case 'everyone':
		INITIAL_PEERS_EVAL = conf.ALL_PEERS
		_init_peers_graph = nx.complete_graph(conf.NODES_COUNT)
	case 'no-one':
		INITIAL_PEERS_EVAL = {}
		_init_peers_graph = nx.empty_graph(conf.NODES_COUNT)
	case 'rgg':
		_init_peers_graph = topogen.rgg(n=conf.NODES_COUNT, start_seed=conf.GLOBAL_RANDOM_SEED)
		
		INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
		INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
	case 'erg':
		_init_peers_graph = topogen.erg(n=conf.NODES_COUNT, start_seed=conf.GLOBAL_RANDOM_SEED)
		
		INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
		INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
	case 'rst':
		_init_peers_graph = topogen.rst(n=conf.NODES_COUNT, seed=conf.GLOBAL_RANDOM_SEED)
		
		INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
		INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
	case 'bag':
		_init_peers_graph = topogen.bag(n=conf.NODES_COUNT, seed=conf.GLOBAL_RANDOM_SEED)
		
		INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
		INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
	case 'wsg':
		_init_peers_graph = topogen.wsg(n=conf.NODES_COUNT, start_seed=conf.GLOBAL_RANDOM_SEED)
		
		INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
		INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
	case _:
		if conf.INITIAL_PEERS.startswith('list-'):
			INITIAL_PEERS_EVAL = set(conf.INITIAL_PEERS.split('-', 1)[1].split(','))
			
			_init_peers_graph: nx.Graph = nx.empty_graph(len(INITIAL_PEERS_EVAL) + 1)
			_init_peers_graph.add_edges_from([(conf.INDEX, conf._node_name_to_index(peer)) for peer in INITIAL_PEERS_EVAL])  # FIXME: protected member usage.
		elif conf.INITIAL_PEERS.startswith('bag-'):
			_m = int(conf.INITIAL_PEERS.split('-', 1)[1])
			
			_init_peers_graph = topogen.bag(n=conf.NODES_COUNT, m=_m, seed=conf.GLOBAL_RANDOM_SEED)
			
			INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
			INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
		elif conf.INITIAL_PEERS.startswith('wsg-'):
			_k = int(conf.INITIAL_PEERS.split('-', 2)[1])
			_p = float(conf.INITIAL_PEERS.split('-', 2)[2])
			
			_init_peers_graph = topogen.wsg(n=conf.NODES_COUNT, k=_k, p=_p, start_seed=conf.GLOBAL_RANDOM_SEED)
			
			INITIAL_PEERS_EVAL = {t[1] for t in _init_peers_graph.edges(conf.INDEX)}
			INITIAL_PEERS_EVAL = {f'n{peer_index}' for peer_index in INITIAL_PEERS_EVAL}
		else:
			assert False

match conf.SIMULATE_LATENCIES.lower():
	case 'none':
		async def _none():
			pass
		
		
		SIMULATE_LATENCIES_EVAL = _none
	case 'initial-peers':
		SIMULATE_LATENCIES_EVAL = functools.partial(latsinit.initial_peers, _init_peers_graph, self_index=conf.INDEX, rnd=Random(conf.GLOBAL_RANDOM_SEED), rnd_range=conf.SIMULATE_LATENCIES_RANDOM_RANGE)
	case 'initial-peers-local':
		SIMULATE_LATENCIES_EVAL = functools.partial(latsinit.initial_peers_local, _init_peers_graph, self_index=conf.INDEX, rnd=Random(conf.GLOBAL_RANDOM_SEED), rnd_range=conf.SIMULATE_LATENCIES_RANDOM_RANGE)
	case 'all-random':
		SIMULATE_LATENCIES_EVAL = functools.partial(latsinit.all_random, conf.ALL_PEERS, rnd=Random(conf.GLOBAL_RANDOM_SEED), rnd_range=conf.SIMULATE_LATENCIES_RANDOM_RANGE)
	case _:
		if conf.SIMULATE_LATENCIES.lower().startswith('list-'):
			_peers_latencies = dict(entry.split('=', maxsplit=1) for entry in conf.SIMULATE_LATENCIES.split('-', 1)[1].split(','))
			_peers_latencies = {peer: int(latency) for peer, latency in _peers_latencies.items()}
			
			SIMULATE_LATENCIES_EVAL = functools.partial(latsinit.list_, _peers_latencies)
		else:
			assert False
