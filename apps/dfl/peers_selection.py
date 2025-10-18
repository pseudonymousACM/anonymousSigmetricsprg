import asyncio
import functools
import logging
import math
import re
from collections import OrderedDict
from datetime import timedelta
from random import Random
from typing import Any, Literal

import numpy as np
import p2psims.extensions.logging as logging_exts
import p2psims.node
import sympy
import tenacity
import torch
from codetiming import Timer
from p2psims import Node as P2pSimsNode
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

import clustering
import conf
import conf_eval
import search_data
import similarity
import singletons
import son_agg
from sync import sync

_logger = logging.getLogger(__name__)


async def _weighted_clustering_decorator(data: dict[str, tuple[torch.Tensor, float]]):
	return await clustering.do({k: t for k, (t, w) in data.items()})


async def _weighted_similarity_decorator(u: tuple[torch.Tensor, float], v: tuple[torch.Tensor, float]) -> float:
	return await similarity.acalc(u[0], v[0])


_preferred_zone_size_factors = sorted(sympy.factorint(conf.PREFERRED_ZONE_SIZE, multiple=True))
_be_initiator_rng = Random(conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)


# TODO: push this to the upstream as a general initiator selection method; identity (int) + modulo.
async def _be_initiator(attempt: int) -> bool:
	# Same as the default baseline `IP + time`, but replacing the dynamic IP with the constant index and the undeterministic time parameter with a node-specific randomness, all to be more deterministic and to enable reproducible runs.
	
	modulo = conf.PREFERRED_ZONE_SIZE
	for factor in _preferred_zone_size_factors[:attempt - 1]:
		modulo //= factor
	
	_logger.debug(f"The modulo for attempt {attempt} is {modulo}.")
	
	return (conf.INDEX + _be_initiator_rng.randint(0, modulo)) % modulo == 0


async def _ensure_init_search(data: Any | None = None, round_: int | None = None) -> Any:
	if round_ is None:
		round_ = singletons.round_
	
	if data is None:
		data = await search_data.get(round_=round_)
	
	if round_ % conf.SEARCH_RENEW_ROUNDS == 0:
		_logger.debug("Renewing the search network...")
		
		if conf.SYNC_SON_INIT:
			await sync('SON construct.')
		
		with Timer(logger=None) as timer:
			session = str(round_)
			last_round = round_ + conf.SEARCH_RENEW_ROUNDS - 1
			
			node_level_multiplex_servicer = p2psims.node.LevelMultiplexingServicer()
			node = await P2pSimsNode(
				data=(data, 1),
				initial_peers_addresses=conf_eval.INITIAL_PEERS_EVAL,
				preferred_zone_size=conf.PREFERRED_ZONE_SIZE,
				no_probe_duration=tenacity.wait_combine(tenacity.wait_fixed(timedelta(seconds=conf.SEARCH_NO_PROBE_DURATION)), tenacity.wait_random()),  # Add one second random jitter to avoid nodes perfectly matching the sleeps.
				similarity=_weighted_similarity_decorator,
				aggregate_data=son_agg.do,
				cluster_data=_weighted_clustering_decorator,
				logger=logging_exts.LoggerAdapter(logging.getLogger('p2psims.node'), extra={'round': round_}, overwrite=False),
				session=session,
				init_grpc_server=False,
				max_zone_size=conf.MAX_ZONE_SIZE,
				advertise_address=conf.NAME,
				level_multiplex_servicer=node_level_multiplex_servicer,
				be_initiator=_be_initiator
			)
			
			await singletons.dfl_node.register_round_cleanup(lambda: node.close(release_stubs=False), round_=last_round)
			
			await singletons.p2psims_sessions_nodes.register_session_servicer(session, node_level_multiplex_servicer)
			await singletons.dfl_node.register_round_cleanup(lambda: singletons.p2psims_sessions_nodes.unregister_session_servicer(session), round_=last_round)
			
			await node.wait_ready()
			
			singletons.p2psims_node = node
		
		_logger.info(f"Search readiness took {timer.last:.2f} seconds.", extra={'type': 'time-search-ready', 'round': round_, 'time-seconds': timer.last})


async def _all_peers_sims(datum: torch.Tensor | None = None, round_: int | None = None, last_son_renewal_round: bool = False) -> dict[str, float]:
	if round_ is None:
		round_ = singletons.round_
	if datum is None:
		datum = await search_data.get(round_=round_, last_son_renewal_round=last_son_renewal_round)
	
	peers_sims = {}
	async with asyncio.TaskGroup() as tg:
		for peer in conf.ALL_PEERS:
			peers_sims[peer] = tg.create_task(search_data.sim(peer, n1_datum=datum, round_=round_, last_son_renewal_round=last_son_renewal_round))
	
	for peer, task in peers_sims.items():
		peers_sims[peer] = await task
	
	peers_sims = OrderedDict(sorted(peers_sims.items(), key=lambda peer_sim: peer_sim[1], reverse=True))
	
	return peers_sims


async def _find_ideal_topk_peers(k: int, round_: int) -> list[str]:
	if conf.DISABLE_CDFL:
		async with asyncio.TaskGroup() as tg:
			tg.create_task(search_data.publish())
			peers_sims = tg.create_task(_all_peers_sims(round_=round_))
			
			peers_sims = await peers_sims
		
		return list(peers_sims.keys())[:k]
	else:
		async with asyncio.TaskGroup() as tg:
			tg.create_task(search_data.publish())
			meta_topks = tg.create_task(singletons.dfl_communicator.receive(from_='cdfl', meta={'type': 'ideal-topk', 'from': conf.NAME, 'round': round_, 'k': k}))
			
			_, topks = await meta_topks
		
		return topks


async def ideal_top_k(k: int, **kwargs):
	round_ = kwargs['round']
	
	topk_peers = await _find_ideal_topk_peers(k=k, round_=round_)
	_logger.info(f"Selected the following peers: {set(topk_peers)}.", extra={'type': 'agg-peers', 'round': round_, 'peers': topk_peers})
	
	return [{'from': peer} for peer in topk_peers]


async def min_similarity_m(m: float, **kwargs):
	round_ = kwargs['round']
	
	data = await search_data.get(round_=round_)
	await _ensure_init_search(data, round_=round_)
	
	_logger.debug("Searching...")
	with Timer(logger=None) as timer:
		res = await singletons.p2psims_node.search((data, 1), min_similarity=m)
	res = {peer.split(':')[0]: sim for peer, sim in res.items()}
	res = OrderedDict(sorted(res.items(), key=lambda it: it[1], reverse=True))
	
	_logger.info(f"Search `min_similarity={m}` (took {timer.last:.2f} seconds), result: {res}.", extra={'type': 'search', 'min_similarity': m, 'result': res, 'round': round_, 'time-seconds': timer.last})
	
	res = set(res.keys())
	res -= {conf.NAME}
	
	_logger.info(f"Selected the following peers: {res}.", extra={'type': 'agg-peers', 'round': round_, 'peers': list(res)})
	
	return [{'from': peer} for peer in res]


async def top_k(min_similarity: float, k: int, sequential_top_k: bool, **kwargs):
	round_ = kwargs['round']
	
	data = await search_data.get(round_=round_)
	await _ensure_init_search(data, round_=round_)
	
	_logger.debug("Searching...")
	with Timer(logger=None) as timer:
		# `k = k + 1`, because it might find itself in the results.
		res = await singletons.p2psims_node.search((data, 1), k=k + 1, min_similarity=min_similarity, sequential_top_k=sequential_top_k)
	res = {peer.split(':')[0]: sim for peer, sim in res.items()}
	res = OrderedDict(sorted(res.items(), key=lambda it: it[1], reverse=True))
	
	_logger.info(f"Search `min_similarity={min_similarity}, k={k + 1}, sequential={sequential_top_k}` (took {timer.last:.2f} seconds), result: {res}.", extra={'type': 'search', 'min_similarity': min_similarity, 'k': k + 1, 'sequential': sequential_top_k, 'result': res, 'round': round_, 'time-seconds': timer.last})
	
	res.pop(conf.NAME, None)
	res = set(t[0] for t in list(res.items())[:k])
	
	_logger.info(f"Selected the following peers: {set(res)}.", extra={'type': 'agg-peers', 'round': round_, 'peers': list(res)})
	
	return [{'from': peer} for peer in res]


async def everyone(**kwargs):
	_logger.info(f"Selected the following peers: {conf.ALL_PEERS}.", extra={'type': 'agg-peers', 'round': kwargs['round'], 'peers': list(conf.ALL_PEERS)})
	
	return [{'from': peer} for peer in conf.ALL_PEERS]


async def random_k(k: int, rnd: Random, **kwargs):
	peers = rnd.sample(sorted(conf.ALL_PEERS), k=k)
	
	_logger.info(f"Selected the following peers: {set(peers)}.", extra={'type': 'agg-peers', 'round': kwargs['round'], 'peers': peers})
	
	return [{'from': peer} for peer in peers]


async def no_one(**kwargs):
	return []


async def initial_peers(**kwargs):
	_logger.info(f"Selected the following peers: {conf_eval.INITIAL_PEERS_EVAL}.", extra={'type': 'agg-peers', 'round': kwargs['round'], 'peers': list(conf_eval.INITIAL_PEERS_EVAL)})
	return [{'from': peer} for peer in conf_eval.INITIAL_PEERS_EVAL]


def _find_max_x_target_with_y_gte_y_target(ir: IsotonicRegression, y_target: float) -> float:
	ts = list(zip(ir.X_thresholds_, ir.y_thresholds_))
	
	x_rightmost, y_rightmost = ts[-1]
	if y_target <= y_rightmost:
		if ir.out_of_bounds == 'clip':
			return math.inf
		else:
			return float(x_rightmost)
	
	for i in range(len(ts) - 1, 0, -1):
		x_right, y_right = ts[i]
		x_left, y_left = ts[i - 1]
		
		if y_right <= y_target <= y_left:
			if y_right == y_left:
				return float(x_right)
			else:
				m = (y_target - y_left) / (y_right - y_left)
				x_target = x_left + m * (x_right - x_left)
				return float(x_target)
	
	if ir.out_of_bounds == 'clip':
		return -math.inf
	else:
		return math.nan


# noinspection PyPep8Naming
def _pred_fit_isotonic_regression(target_k: int, M: int, history: list[tuple[float, int, int]], decay: Literal['lin', 'none'] = 'lin', active_decay: bool = False) -> tuple[float, IsotonicRegression]:
	ir = IsotonicRegression(increasing=False, out_of_bounds='clip')
	
	if len(history) == 0:
		min_sims, ks, rounds = tuple(), tuple(), tuple()
	else:
		min_sims, ks, rounds = tuple(zip(*history))
	
	match decay:
		case 'none':
			samples_weights = None
		case 'lin':
			rounds_count = len(np.unique(rounds))
			
			if rounds_count == 0:
				samples_weights = tuple()
			else:
				# TODO: avoid 1, just like 0?
				rounds_weights = np.linspace(0, 1, rounds_count + 1)
				rounds_weights = rounds_weights[1:]
				samples_weights = tuple(rounds_weights[list(np.array(rounds) - np.min(rounds))].tolist())
		
		# TODO: add an alternative exponential decay type.
		case _:
			# noinspection PyUnreachableCode
			assert False
	
	if active_decay and len(samples_weights) > 0:
		# Assert `samples_weights` are in range [0, 1].
		
		decay_weights = 1 - np.array(samples_weights)
		decay_min_sims = np.array(min_sims)
		decay_ks = -M * decay_min_sims + M
		
		min_sims += tuple(decay_min_sims.tolist())
		ks += tuple(decay_ks.tolist())
		samples_weights += tuple(decay_weights.tolist())
	
	if samples_weights is not None and len(samples_weights) > 0:
		w_inf = np.max(samples_weights) * len(samples_weights)
	else:
		w_inf = 1
	
	ir.fit(min_sims + (0, 1), ks + (M, 0), (samples_weights + (w_inf, w_inf)) if samples_weights is not None else None)
	
	pred_min_sim = _find_max_x_target_with_y_gte_y_target(ir, target_k)
	if math.isinf(pred_min_sim):
		if pred_min_sim > 0:
			pred_min_sim = 1
		else:
			# Assume M is large enough for this to not happen.
			# An active workaround can be increasing the M value to at-least k and retrying. TODO.
			assert False
	
	return pred_min_sim, ir


_past_min_sims_ks_rounds = []


# noinspection PyPep8Naming
async def isotonic_parallel_top_k(k: int, M: int, max_hist_len: int | None = None, decay: Literal['lin', 'none'] = 'lin', augment: bool = False, active_decay: bool = False, **kwargs):
	global _past_min_sims_ks_rounds
	
	round_ = kwargs['round']
	
	data = await search_data.get(round_=round_)
	await _ensure_init_search(data, round_=round_)
	
	if len(_past_min_sims_ks_rounds) >= 2:
		min_round, max_round = _past_min_sims_ks_rounds[0][2], _past_min_sims_ks_rounds[-1][2]
		if max_hist_len is not None and (max_round - min_round + 1) > max_hist_len:
			_past_min_sims_ks_rounds = [skr for skr in _past_min_sims_ks_rounds if skr[2] > min_round]
	
	pred_min_sim, ir = _pred_fit_isotonic_regression(target_k=k, M=M, history=_past_min_sims_ks_rounds, decay=decay, active_decay=active_decay)
	
	_logger.debug("Searching...")
	with Timer(logger=None) as timer:
		res = await singletons.p2psims_node.search((data, 1), min_similarity=pred_min_sim)
	res = {peer.split(':')[0]: sim for peer, sim in res.items()}
	
	res = dict(sorted(res.items(), key=lambda it: it[1], reverse=True))
	_logger.info(f"Search `min_similarity={pred_min_sim}` (took {timer.last:.2f} seconds), result: {res}.", extra={'type': 'search', 'min_similarity': pred_min_sim, 'result': res, 'round': round_, 'time-seconds': timer.last, 'history': _past_min_sims_ks_rounds, 'target_k': k, 'isotonic_xys': list(zip(ir.X_thresholds_, ir.y_thresholds_))})
	
	res.pop(conf.NAME, None)
	_past_min_sims_ks_rounds.append((pred_min_sim, len(res), round_))
	
	res = list(res.items())
	if augment:
		for i, (_, sim) in enumerate(res):
			_past_min_sims_ks_rounds.append((sim, i + 1, round_))
	
	res = res[:k]
	res = set(t[0] for t in res)
	
	_logger.info(f"Selected the following peers: {res}.", extra={'type': 'agg-peers', 'round': round_, 'peers': list(res)})
	return [{'from': peer} for peer in res]


def _pred_fit_lin_reg(target_k: int, history: list[tuple[float, int, int]], decay: Literal['lin', 'none'] = 'lin') -> tuple[float, float]:
	if len(history) == 0:
		raise ValueError('No history.')
	
	min_sims, ks, rounds = tuple(zip(*history))
	
	match decay:
		case 'none':
			samples_weights = None
		case 'lin':
			rounds_count = len(np.unique(rounds))
			
			if rounds_count == 0:
				samples_weights = tuple()
			else:
				# TODO: avoid 1, just like 0?
				rounds_weights = np.linspace(0, 1, rounds_count + 1)
				rounds_weights = rounds_weights[1:]
				samples_weights = tuple(rounds_weights[list(np.array(rounds) - np.min(rounds))].tolist())
		
		# TODO: add an alternative exponential decay type.
		case _:
			# noinspection PyUnreachableCode
			assert False
	
	min_sims_transformed = np.array(min_sims) - 1
	min_sims_transformed = min_sims_transformed.reshape(-1, 1)
	
	lr = LinearRegression(fit_intercept=False)
	lr.fit(min_sims_transformed, ks, samples_weights)
	
	m = lr.coef_[0]
	if m == 0:
		pred_min_sim = 0
	else:
		pred_min_sim = (target_k / m) + 1
	
	return pred_min_sim, m


async def par_topk_lin_reg(target_k: int, init_min_sim: float, max_hist_len: int | None = None, decay: Literal['lin', 'none'] = 'lin', **kwargs):
	global _past_min_sims_ks_rounds
	
	round_ = kwargs['round']
	
	data = await search_data.get(round_=round_)
	await _ensure_init_search(data, round_=round_)
	
	if len(_past_min_sims_ks_rounds) >= 2:
		min_round, max_round = _past_min_sims_ks_rounds[0][2], _past_min_sims_ks_rounds[-1][2]
		if max_hist_len is not None and (max_round - min_round + 1) > max_hist_len:
			_past_min_sims_ks_rounds = [skr for skr in _past_min_sims_ks_rounds if skr[2] > min_round]
	
	if len(_past_min_sims_ks_rounds) == 0:
		pred_min_sim, m = init_min_sim, -math.inf
	else:
		pred_min_sim, m = _pred_fit_lin_reg(target_k=target_k, history=_past_min_sims_ks_rounds, decay=decay)
	
	_logger.debug("Searching...")
	with Timer(logger=None) as timer:
		res = await singletons.p2psims_node.search((data, 1), min_similarity=pred_min_sim)
	res = {peer.split(':')[0]: sim for peer, sim in res.items()}
	
	res = dict(sorted(res.items(), key=lambda it: it[1], reverse=True))
	_logger.info(f"Search `min_similarity={pred_min_sim}` (took {timer.last:.2f} seconds), result: {res}.", extra={'type': 'search', 'min_similarity': pred_min_sim, 'result': res, 'round': round_, 'time-seconds': timer.last, 'history': _past_min_sims_ks_rounds, 'target_k': target_k, 'lin_reg_m': m})
	
	res.pop(conf.NAME, None)
	_past_min_sims_ks_rounds.append((pred_min_sim, len(res), round_))
	
	res = list(res.items())
	res = res[:target_k]
	res = set(t[0] for t in res)
	
	_logger.info(f"Selected the following peers: {res}.", extra={'type': 'agg-peers', 'round': round_, 'peers': list(res)})
	return [{'from': peer} for peer in res]


async def _par_topk_idealth(target_k: int, datum: torch.Tensor, round_: int) -> float:
	if conf.DISABLE_CDFL:
		async with asyncio.TaskGroup() as tg:
			if round_ % conf.SEARCH_RENEW_ROUNDS == 0:  # SON renewal round.
				tg.create_task(search_data.publish())
			peers_sims = tg.create_task(_all_peers_sims(datum=datum, round_=round_, last_son_renewal_round=True))
			
			peers_sims = await peers_sims
		
		sims = list(peers_sims.values())
		ideal_min_sim = (
				                (sims[target_k - 1] if target_k > 0 else 1)
				                + (sims[target_k] if target_k < (conf.NODES_COUNT - 1) else 0)
		                ) / 2
	else:
		async with asyncio.TaskGroup() as tg:
			tg.create_task(search_data.publish())
			meta_ideal_min_sim = tg.create_task(singletons.dfl_communicator.receive(from_='cdfl', meta={'type': 'par-topk-idealth', 'from': conf.NAME, 'round': round_, 'k': target_k}))
			
			_, ideal_min_sim = await meta_ideal_min_sim
	
	# noinspection PyUnboundLocalVariable
	return ideal_min_sim


async def par_topk_ideal_th(target_k: int, **kwargs):
	round_ = kwargs['round']
	datum = await search_data.get(round_=round_)
	
	async with asyncio.TaskGroup() as tg:
		tg.create_task(_ensure_init_search(data=datum, round_=round_))
		ideal_min_sim = tg.create_task(_par_topk_idealth(target_k, datum, round_))
		
		ideal_min_sim = await ideal_min_sim
	
	_logger.debug("Searching...")
	with Timer(logger=None) as timer:
		res = await singletons.p2psims_node.search((datum, 1), min_similarity=ideal_min_sim)
	
	res = {peer.split(':')[0]: sim for peer, sim in res.items()}
	res = dict(sorted(res.items(), key=lambda it: it[1], reverse=True))
	
	_logger.info(f"Search `min_similarity={ideal_min_sim}` (took {timer.last:.2f} seconds), result: {res}.", extra={'type': 'search', 'min_similarity': ideal_min_sim, 'result': res, 'round': round_, 'time-seconds': timer.last, 'target_k': target_k})
	
	res.pop(conf.NAME, None)
	res = set(list(res.keys())[:target_k])
	
	_logger.info(f"Selected the following peers: {res}.", extra={'type': 'agg-peers', 'round': round_, 'peers': list(res)})
	return [{'from': peer} for peer in res]


match conf.PEERS_SELECTION:
	case 'everyone':
		_f = everyone
	case 'no-one':
		_f = no_one
	case 'initial-peers':
		_f = initial_peers
	case _:
		if conf.PEERS_SELECTION.startswith('ideal-top-'):
			_k = int(conf.PEERS_SELECTION.split('-', 2)[-1])
			_f = functools.partial(ideal_top_k, k=_k)
		elif conf.PEERS_SELECTION.startswith('min-similarity-'):
			_m = float(conf.PEERS_SELECTION.split('-', 2)[-1])
			_f = functools.partial(min_similarity_m, m=_m)
		elif conf.PEERS_SELECTION.startswith('uniform-random-'):
			_k = int(conf.PEERS_SELECTION.split('-', 2)[-1])
			_f = functools.partial(random_k, k=_k, rnd=Random(conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX))
		elif conf.PEERS_SELECTION.startswith('top-'):
			_match = re.fullmatch(r'^top-(?P<k>\d+)-(?P<m>\d*\.\d+)(-seq)?$', conf.PEERS_SELECTION)
			_k, _m = int(_match.group('k')), float(_match.group('m'))
			_seq = conf.PEERS_SELECTION.endswith('-seq')
			
			_f = functools.partial(top_k, k=_k, min_similarity=_m, sequential_top_k=_seq)
		elif conf.PEERS_SELECTION.startswith('parallel-topk-isotonic-'):
			_kvs = conf.PEERS_SELECTION.removeprefix('parallel-topk-isotonic-')
			_kvs = dict[str, str](kv.split('=', 1) for kv in _kvs.split('-'))
			
			_k = int(_kvs['k'])
			_M = int(_kvs['m'])
			_h = _kvs.get('h', None)
			if _h is not None:
				_h = int(_h)
				assert _h >= 1
			_decay = _kvs.get('decay', 'lin')
			_augment = _kvs.get('augment', None)
			if _augment is not None:
				_augment = _augment.lower() in ('true', 'y', 'yes', '1')
			else:
				_augment = False
			_active_decay = _kvs.get('adecay', None)
			if _active_decay is not None:
				_active_decay = _active_decay.lower() in ('true', 'y', 'yes', '1')
			else:
				_active_decay = False
			
			assert 0 <= _k <= _M
			assert _decay in ('none', 'lin')
			assert not _active_decay or (_decay != 'none')
			
			_f = functools.partial(isotonic_parallel_top_k, k=_k, M=_M, max_hist_len=_h, decay=_decay, augment=_augment, active_decay=_active_decay)
		elif conf.PEERS_SELECTION.startswith('par-topk-lin-reg-'):
			_kvs = conf.PEERS_SELECTION.removeprefix('par-topk-lin-reg-')
			_kvs = dict[str, str](kv.split('=', 1) for kv in _kvs.split('-'))
			
			_k = int(_kvs['k'])
			assert _k >= 1
			
			_init = float(_kvs['init'])
			assert 0 <= _init < 1
			
			_hist = _kvs.get('hist', None)
			if _hist is not None:
				_hist = int(_hist)
				assert _hist >= 1
			
			_decay = _kvs.get('decay', 'lin')
			assert _decay in ('none', 'lin')
			
			_f = functools.partial(par_topk_lin_reg, target_k=_k, init_min_sim=_init, max_hist_len=_hist, decay=_decay)
			
			del _kvs, _k, _init, _hist, _decay
		elif conf.PEERS_SELECTION.startswith('par-topk-ideal-th-'):
			_kvs = conf.PEERS_SELECTION.removeprefix('par-topk-ideal-th-')
			_kvs = dict[str, str](kv.split('=', 1) for kv in _kvs.split('-'))
			
			_k = int(_kvs['k'])
			assert 0 <= _k <= (conf.NODES_COUNT - 1)  # -1 because cannot select self.
			
			_f = functools.partial(par_topk_ideal_th, target_k=_k)
			
			del _kvs, _k
		else:
			assert False


async def do(**kwargs):
	return await _f(**kwargs)
