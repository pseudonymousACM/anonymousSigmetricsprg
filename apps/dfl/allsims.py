import asyncio
import itertools
import logging
from random import Random

from dfl.extensions.frozenset import pretty_format as pretty_format_frozenset

import conf
import search_data
import singletons

_logger = logging.getLogger(__name__)

if conf.MONITOR_SIMILARITIES == 'split':
	_rng = Random(conf.GLOBAL_RANDOM_SEED)
	_all_nodes_shuffle_container = conf.ALL_NODES_LIST.copy()


def _step_split_state() -> None:
	_rng.shuffle(_all_nodes_shuffle_container)


def _step_none_state() -> None:
	pass  # NOP.


def _infer_my_next_peers() -> set[str]:
	_step_split_state()
	
	my_peers = set[str]()
	
	for ((nai, na), (nbi, nb)) in itertools.combinations(enumerate(_all_nodes_shuffle_container), 2):
		if conf.NAME not in {na, nb}:
			continue
		
		s = nai + nbi
		if s % 2 == 0 and na == conf.NAME:
			my_peers.add(na if nb == conf.NAME else nb)
		elif s % 2 != 0 and nb == conf.NAME:
			my_peers.add(na if nb == conf.NAME else nb)
	
	return my_peers


# TODO: optimize by reusing the computed similarities between the normal similarities and the SON similarities when the current round is a SON renewal round.

async def _step_split() -> None:
	my_peers = _infer_my_next_peers()
	_logger.debug(f"My peers: {my_peers}.")
	
	peers_sims = dict[str, asyncio.Task | float]()
	if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
		nodes_son_sims = dict[tuple[str, str], float | asyncio.Task]()  # Assume the right side is the SON's.
	
	self_datum = await search_data.get(force_comm=True)
	
	async with asyncio.TaskGroup() as tg:
		_logger.debug("Computing similarities...")
		
		for peer in my_peers:
			peers_sims[peer] = tg.create_task(search_data.sim(peer, n1_datum=self_datum, force_comm=True))
		
		if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
			for node in conf.ALL_NODES:
				# noinspection PyUnboundLocalVariable
				nodes_son_sims[node] = tg.create_task(search_data.sim(node, n1_datum=self_datum, n0_last_son_renewal_round=True, force_comm=True))
	
	for peer, task in peers_sims.items():
		peers_sims[peer] = await task
	if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
		for node, task in nodes_son_sims.items():
			nodes_son_sims[node] = await task
	
	# FIXME: truncate this log's message when there are too many similarities; e.g., show at most 7 similarities.
	
	sims = {frozenset((conf.NAME, peer)): sim for peer, sim in peers_sims.items()}
	_logger.info(f"Similarities: {{{', '.join([f"{pretty_format_frozenset(fs)}: {sim:.2f}" for fs, sim in sims.items()])}}}.", extra={'type': 'similarities', 'round': singletons.round_, 'similarities': tuple(((n0, n1), sim) for (n0, n1), sim in sims.items())})
	if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
		_logger.info(f"SON Similarities: {{{', '.join([f"({n0}, {n1}): {sim:.2f}" for (n0, n1), sim in nodes_son_sims.items()])}}}.", extra={'type': 'son-sims', 'round': singletons.round_, 'sims': tuple(((n0, n1), sim) for (n0, n1), sim in nodes_son_sims.items())})


async def _step_n0() -> None:
	nodes_sims = dict[frozenset[str], float | asyncio.Task]()
	if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
		nodes_son_sims = dict[tuple[str, str], float | asyncio.Task]()  # Assume the right side is the SON's.
	
	async with asyncio.TaskGroup() as tg:
		_logger.debug("Computing similarities...")
		
		for n0, n1 in itertools.combinations(conf.ALL_NODES, r=2):
			nodes_sims[frozenset((n0, n1))] = tg.create_task(search_data.sim(n0, n1, force_comm=True))
		
		if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
			for n0, n1 in itertools.product(conf.ALL_NODES, repeat=2):
				# noinspection PyUnboundLocalVariable
				nodes_son_sims[(n0, n1)] = tg.create_task(search_data.sim(n0, n1, n1_last_son_renewal_round=True, force_comm=True))
	
	for node, task in nodes_sims.items():
		nodes_sims[node] = await task
	if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
		for node, task in nodes_son_sims.items():
			nodes_son_sims[node] = await task
	
	_logger.info(f"Similarities: {{{', '.join([f"{pretty_format_frozenset(fs)}: {sim:.2f}" for fs, sim in itertools.islice(nodes_sims.items(), 7)])}{'}' if len(nodes_sims) <= 7 else '...'}.", extra={'type': 'similarities', 'round': singletons.round_, 'similarities': tuple(((n0, n1), sim) for (n0, n1), sim in nodes_sims.items())})
	if conf.SEARCH_FUL_PEERS_SELECTION and singletons.round_ >= 0:
		_logger.info(f"SON Similarities: {{{', '.join([f"({n0}, {n1}): {sim:.2f}" for (n0, n1), sim in itertools.islice(nodes_son_sims.items(), 7)])}{'}' if len(nodes_son_sims) <= 7 else '...'}.", extra={'type': 'son-sims', 'round': singletons.round_, 'sims': tuple(((n0, n1), sim) for (n0, n1), sim in nodes_son_sims.items())})


async def _step_none() -> None:
	pass  # NOP.


match conf.MONITOR_SIMILARITIES:
	case 'n0':
		if conf.INDEX == 0:
			step = _step_n0
			step_state = _step_none_state
		else:
			step = _step_none
			step_state = _step_none_state
	case 'split':
		step = _step_split
		step_state = _step_split_state
	case 'none':
		step = _step_none
		step_state = _step_none_state
	case _:
		assert False
