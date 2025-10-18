import asyncio
import logging
from typing import Any, override

import numpy as np
from dfl import Meta
from dfl.communication import LocalPublishingCommunicator
from dfl.node.torch import Node

import conf
import search_data

_logger = logging.getLogger(__name__)


class CDflNode(Node):
	def __init__(self, logger: logging.Logger | logging.LoggerAdapter | None = None) -> None:
		if logger is None:
			logger = _logger
		
		# noinspection PyTypeChecker
		super().__init__(
			logger=logger,
			cleanup_rounds_age=conf.ROUNDS_AGE,
			torch_context=None,
			communicator=None,
			select_neighbors=None,
			param_agg_factory=None,
			segmenter=None,
			pull_early=False,
			extra_publish_meta=None,
			post_train=None
		)
		
		self._local_comm = LocalPublishingCommunicator()
	
	async def receive_handler(self, meta: Meta | None) -> tuple[Meta | None, Any]:
		type_ = meta['type']
		
		match type_:
			case 'par-topk-idealth':
				round_ = meta['round']
				from_ = meta['from']
				k = meta['k']
				
				ret_meta, sims = await self._local_comm.subscribe({'type': f'{type_}-sims', 'round': round_, 'for': from_})
				
				ideal_min_sim = (
						                (sims[k - 1] if k > 0 else 1)
						                + (sims[k] if k < (conf.NODES_COUNT - 1) else 0)
				                ) / 2
				
				return None, ideal_min_sim
			case 'ideal-topk':
				round_ = meta['round']
				from_ = meta['from']
				idx = conf._node_name_to_index(from_)  # FIXME: protected member usage.
				k = meta['k']
				
				ret_meta, sims = await self._local_comm.subscribe({'type': 'ideal-topk', 'round': round_})
				topks = np.argpartition(sims[idx], -k)[-k:].tolist()
				topks = [f'n{idx}' for idx in topks]
				
				return ret_meta | {'for': from_, 'k': k}, topks
			case _:
				raise ValueError(f"Unexpected type {type_}.")
	
	@override
	async def step_round(self, epochs: int | None = None, cleanup: bool = True):
		assert epochs is None
		
		self._logger.debug(f"Stepping round #{self.round_}...")
		
		if conf.PEERS_SELECTION.startswith('par-topk-ideal-th-'):
			await self._par_topk_ideal_th()
		elif conf.PEERS_SELECTION.startswith('ideal-top-'):
			await self._ideal_topk()
		
		self.round_ += 1
		if cleanup:
			await self.cleanup()
		
		self._logger.debug(f"Stepped round #{self.round_ - 1}.")
	
	async def _par_topk_ideal_th(self) -> None:
		self._logger.debug("Computing SON par. top-k ideal min. sim. th. for all the nodes...")
		
		async with asyncio.TaskGroup() as tg:
			for node in conf.ALL_NODES:
				tg.create_task(self._par_topk_ideal_th_for(node))
		
		self._logger.debug("Computed SON par. top-k ideal min. sim. th. for all the nodes.")
	
	async def _par_topk_ideal_th_for(self, for_: str) -> None:
		peers_son_sims = dict[str, float | asyncio.Task[float]]()
		async with asyncio.TaskGroup() as tg:
			for peer in conf.ALL_NODES:
				if peer == for_:
					continue
				
				peers_son_sims[peer] = tg.create_task(search_data.sim(n0=for_, n1=peer, n1_last_son_renewal_round=True, round_=self.round_, force_comm=True))
			
			for peer, task in peers_son_sims.items():
				peers_son_sims[peer] = await task
		
		sims = list(peers_son_sims.values())
		sims.sort(reverse=True)
		
		meta = {'type': 'par-topk-idealth-sims', 'round': self.round_, 'for': for_}
		await self._local_comm.publish(meta, sims)
		await self.register_round_cleanup(lambda: self._local_comm.unpublish(meta))
	
	async def _ideal_topk(self) -> None:
		self._logger.debug("Computing pairwise similarities for ideal top-k...")
		
		sims = np.zeros((conf.NODES_COUNT, conf.NODES_COUNT), dtype=object)
		np.fill_diagonal(sims, -np.inf)  # Avoids selecting self.
		
		upper_is, upper_js = np.triu_indices_from(sims, k=1)
		lower_is, lower_js = upper_js, upper_is
		
		async with asyncio.TaskGroup() as tg:
			for i, j in zip(lower_is, lower_js):
				sims[i][j] = tg.create_task(search_data.sim(n0=f'n{i}', n1=f'n{j}', round_=self.round_, force_comm=True))
			
			for i, j in zip(lower_is, lower_js):
				sims[i][j] = await sims[i][j]
		
		sims = sims.astype(float)  # TODO: do this in-place if possible.
		sims[upper_is, upper_js] = sims[lower_is, lower_js]
		
		meta = {'type': 'ideal-topk', 'round': self.round_}
		await self._local_comm.publish(meta, sims)
		await self.register_round_cleanup(lambda: self._local_comm.unpublish(meta))
		
		self._logger.debug("Computed pairwise similarities for ideal top-k.")
	
	@override
	async def pull(self):
		raise NotImplementedError
	
	@override
	async def train(self, epochs: int | None = 1):
		raise NotImplementedError
	
	@override
	async def post_train(self) -> None:
		raise NotImplementedError
	
	@override
	async def train_eval(self):
		raise NotImplementedError
	
	@override
	async def publish(self):
		raise NotImplementedError
	
	@override
	async def aggregate(self, pull=True):
		raise NotImplementedError
	
	@override
	async def test_eval(self, log_pre_agg: bool = False):
		raise NotImplementedError
