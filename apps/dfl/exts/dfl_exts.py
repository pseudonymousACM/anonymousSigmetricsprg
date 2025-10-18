import asyncio
import io
import logging
from typing import Callable, Awaitable, Any, Iterable, override, Literal

import torch
from dfl import Communicator, Meta, Aggregator
from dfl.communication import LocalPublishingCommunicator
from dfl.extensions.frozendict import frozendict
from dfl.extensions.torch import Context as TorchContext
from dfl.node.torch import Node


async def _araise(*args, **kwargs):
	raise NotImplementedError


def _raise(*args, **kwargs):
	raise NotImplementedError


class Client(Node):
	def __init__(
			self,
			torch_context: TorchContext,
			communicator: Communicator,
			server: str,
			logger: logging.Logger | None = None,
			extra_publish_meta: dict | None = None,
			post_train: Callable[[], Awaitable[None]] | None = None
	):
		super().__init__(
			torch_context=torch_context,
			communicator=communicator,
			select_neighbors=_araise,
			param_agg_factory=_raise,
			logger=logger,
			segmenter=None,
			pull_early=False,
			cleanup_rounds_age=0,  # TODO: what is the point of public cleanup when the user can not configure this?
			extra_publish_meta=extra_publish_meta,
			post_train=post_train
		)
		
		self._server = server
		
		self._local_comm = LocalPublishingCommunicator()
	
	async def receive_handler(self, meta: Meta | None) -> (Meta | None, Any):
		assert meta['from'] == self._server
		# assert meta['round'] == self.round_ # TODO: should this be?
		
		meta, data = await self._local_comm.subscribe({'round': meta['round']})
		
		meta = meta | {'from': self._communicator.name} | self._extra_publish_meta
		
		return meta, data
	
	async def pull(self) -> None:
		self._logger.debug('Pulling...')
		
		_, data = await self._communicator.receive(self._server, {'round': self.round_, 'for': self._communicator.name, 'from': self._communicator.name})
		data = torch.load(io.BytesIO(data), weights_only=True)
		
		# noinspection PyTypeChecker
		missing_keys, unexpected_keys = await asyncio.to_thread(self._torch_context.model.load_state_dict, data)
		# There should not be any unexpected key; nor missing key, as the self model would tend to all keys.
		if len(unexpected_keys) > 0 or len(missing_keys) > 0:
			self._logger.error(f"Loading the model's state-dict reported some abnormal keys; unexpected keys: {', '.join(unexpected_keys)}; missing keys: {', '.join(missing_keys)}.")
		
		self._logger.debug('Pulled.')
	
	async def publish(self):
		sd = await asyncio.to_thread(self._torch_context.model.state_dict)
		buff = io.BytesIO()
		torch.save(sd, buff)
		
		meta, data = {'round': self.round_}, buff.getvalue()
		await self._local_comm.publish(meta, data)
		await self.register_round_cleanup(lambda: self._local_comm.unpublish(meta))
		
		self._logger.debug(f"Published {meta}.")
	
	async def step_round(self, epochs: int | None = 1, cleanup: bool = True) -> None:
		self._logger.debug(f"Stepping round #{self.round_}...")
		
		await self.train(epochs)
		async with asyncio.TaskGroup() as tg:
			tg.create_task(self.train_eval())
			tg.create_task(self.test_eval(log_pre_agg=True))
			tg.create_task(self.post_train())
			tg.create_task(self.publish())
		
		await self.pull()
		
		# TODO: can do early cleanup of the published data here.
		
		await self.test_eval()
		
		self.round_ += 1
		if cleanup:
			await self.cleanup()
		
		self._logger.debug(f"Stepped round #{self.round_ - 1}.")
	
	async def aggregate(self, pull=True):
		raise NotImplementedError


class Server(Node):
	def __init__(
			self,
			communicator: Communicator,
			agg_factory: Callable[..., Aggregator],
			clients: set[str],
			logger: logging.Logger | None = None,
			agg_type: Literal['clustered', 'all', 'selective'] = 'all',
			name: str | None = None
	):
		# noinspection PyTypeChecker
		super().__init__(
			torch_context=None,
			communicator=communicator,
			select_neighbors=_araise,
			param_agg_factory=_raise,
			logger=logger,
			segmenter=None,
			pull_early=False,
			cleanup_rounds_age=1,  # TODO: what is the point of public cleanup when the user can not configure this?
			extra_publish_meta=None,
			post_train=None
		)
		
		self._agg_factory = agg_factory
		self._clients = clients
		self._agg_type = agg_type
		
		if name is None:
			name = self._communicator.name
		self._name = name
		
		self._local_comm = LocalPublishingCommunicator()
	
	async def receive_handler(self, meta: Meta | None) -> (Meta | None, Any):
		assert meta['from'] in self._clients
		assert meta['round'] in (self.round_ - 1, self.round_, self.round_ + 1)
		
		res_meta, data = await self._local_comm.subscribe({'round': meta['round']})
		
		res_meta = res_meta | {'from': self._name}
		
		if self._agg_type in ('clustered', 'selective'):
			data = data[meta['for']]
		
		buff = io.BytesIO()
		torch.save(data, buff)
		data = buff.getvalue()
		
		return res_meta, data
	
	async def _select_neighbors(self, break_segments=False, **kwargs) -> Iterable[Meta]:
		assert not break_segments
		
		return [{'from': client, 'round': kwargs['round']} for client in self._clients]
	
	async def _pull_one(self, meta: Meta) -> None:
		frozen_meta = frozendict(meta)
		self._logger.debug(f"Pulling {frozen_meta}...")
		
		from_ = meta['from']
		meta, data = await self._communicator.receive(from_, {'from': self._name, 'round': self.round_})
		
		data = torch.load(io.BytesIO(data), weights_only=True)
		await (await self._agg()).add(meta, data)
		
		self._logger.debug(f"Pulled {frozen_meta}.")
	
	async def aggregate(self, pull=False):
		assert not pull
		
		self._logger.debug("Aggregating...")
		
		res = await (await self._agg()).aggregate()
		self._agg_ = None
		
		# TODO: Used the DFL-style aggregation peers logs for agility; should probably use a CFL-specific log for the matter.
		match self._agg_type:
			case 'all':
				list_clients = list(self._clients)
				for client in self._clients:
					self._logger.info(f"Client {client} aggregated with the following peers: {self._clients}.", extra={'type': 'agg-peers', 'node': client, 'round': self.round_, 'peers': list_clients})
			case 'clustered':
				res = res.items() if isinstance(res, dict) else res
				
				new_res = dict()
				for (clients, res_data) in res:
					list_clients = list(clients)
					
					for client in clients:
						self._logger.info(f"Client {client} aggregated with the following peers: {clients}.", extra={'type': 'agg-peers', 'node': client, 'round': self.round_, 'peers': list_clients})
						
						new_res[client] = res_data
				
				res = new_res
			case 'selective':
				res = res.items() if isinstance(res, dict) else res
				
				new_res = dict()
				for client, (selections, res_data) in res:
					self._logger.info(f"Client {client} aggregated with the following peers: {selections}.", extra={'type': 'agg-peers', 'node': client, 'round': self.round_, 'peers': list(selections)})
					
					new_res[client] = res_data
				
				res = new_res
			case _:
				assert False
		
		meta = {'round': self.round_}
		await self._local_comm.publish(meta, res)
		self._logger.debug(f"Published {meta}.")
		
		await self.register_round_cleanup(lambda: self._local_comm.unpublish(meta))
		
		self._logger.debug("Aggregated.")
	
	async def step_round(self, epochs: int | None = 0, cleanup: bool = True):
		assert epochs == 0
		
		self._logger.debug(f"Stepping round #{self.round_}...")
		
		await self.pull()
		
		# TODO: can do early cleanup of the published data here.
		
		await self.aggregate()
		
		self.round_ += 1
		if cleanup:
			await self.cleanup()
		
		self._logger.debug(f"Stepped round #{self.round_ - 1}.")
	
	async def publish(self):
		raise NotImplementedError
	
	async def train_eval(self):
		raise NotImplementedError
	
	async def test_eval(self):
		raise NotImplementedError
	
	async def post_train(self) -> None:
		raise NotImplementedError
	
	async def train(self, epochs: int | None = 1):
		raise NotImplementedError


class DataClusteringAggregator(Aggregator):
	def __init__(
			self,
			intra_cluster_agg_factory: Callable[[], Aggregator],
			cluster: Callable[[dict[str, Any]], Awaitable[dict[str, set[str]]]],
			key: str
	):
		self._intra_cluster_agg_factory = intra_cluster_agg_factory
		self._cluster = cluster
		self._key = key
		
		self._data = dict[str, tuple[Meta | None, Any]]()
		self._data_lock = asyncio.Lock()
	
	async def add(self, meta: Meta | None = None, data=None) -> None:
		key = meta[self._key]
		async with self._data_lock:
			assert key not in self._data
			self._data[key] = (meta, data)
	
	async def aggregate(self) -> Iterable[tuple[set[Any], Any]]:
		async with self._data_lock:
			data = {k: data for k, (meta, data) in self._data.items()}
		clusters = await self._cluster(data)
		
		res = list[tuple[set[Any], Any]]()
		for _, clients in clusters.items():
			agg = self._intra_cluster_agg_factory()
			
			for client in clients:
				meta, data = self._data[client]
				await agg.add(meta, data)
			
			res.append((clients, await agg.aggregate()))
		
		return res


async def _tensor_to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
	# noinspection PyTypeChecker
	return await asyncio.to_thread(t.to, device=device, non_blocking=True)


async def _state_dict_to_device(state_dict: dict[str, torch.Tensor], device: torch.device, inplace: bool = False) -> dict[str, torch.Tensor]:
	if not inplace:
		state_dict = state_dict.copy()
	
	async with asyncio.TaskGroup() as tg:
		for k, tensor in state_dict.items():
			# noinspection PyTypeChecker
			state_dict[k] = tg.create_task(_tensor_to_device(tensor, device))
		
		for k, task in state_dict.items():
			# noinspection PyUnresolvedReferences
			state_dict[k] = await task
	
	return state_dict


class StateDictToDeviceDecoratingAggregator(Aggregator):
	def __init__(self, device, agg: Aggregator, inplace: bool = False):
		super().__init__()
		self._device = device
		self._agg = agg
		self._inplace = inplace
	
	@override
	async def add(self, meta: Meta | None = None, data=None) -> None:
		data = await _state_dict_to_device(data, device=self._device, inplace=self._inplace)
		await self._agg.add(meta, data)
	
	@override
	async def aggregate(self) -> Any:
		return await self._agg.aggregate()
