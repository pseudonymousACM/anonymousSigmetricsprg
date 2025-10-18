import conf
import logs  # Imported for its side effects.
import repro  # Imported for its side effects.

if conf.TRACE_RPCS:
	import otel
	
	otel.instrument()

import asyncio
import gc
import json
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import timedelta
from os import environ
from typing import Any

import grpc
import p2psims.node
import torch
# noinspection PyUnresolvedReferences
from data import train_data_loader, test_data_loader, train_data, train_eval_data_loader
from dfl import Meta
from dfl.aggregation.torch import TensorToDeviceDecoratingAggregator, WeightedAvgTensorsAggregator
from dfl.communication.caching import StrongSubscribeCachingCommunicatorDecorator
from dfl.communication.encoders import ChainEncoder, PickleBytesEncoder
from dfl.communication.encoders.torch import TensorBytesEncoder
from dfl.communication.grpc import GrpcCommunicator
from dfl.communication.logging import LoggingCommunicatorDecorator
from dfl.node.torch import Node as DflNode
from p2psims import Node as P2PSimsNode
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR

import allsims
import cfl
import conf_eval
import optim
import peers_selection
import search_data
import singletons
import sync
from cdfl import CDflNode
from exts.dfl_exts import Server as CflServer, Client as CflClient
from model import model
from train import TorchContext

_environ_dict = dict(environ)
logging.info(f"Environment: {_environ_dict}.", extra={'type': 'env', 'env': _environ_dict})

P2PSimsNode.grpc_options |= {
	'grpc.service_config': json.dumps({
		'methodConfig': [
			{
				'name': [{}],
				'retryPolicy': {
					'maxAttempts': 5,
					'initialBackoff': '1s',
					'maxBackoff': '7s',
					'backoffMultiplier': 2,
					'retryableStatusCodes': ['UNAVAILABLE'],
				},
			}
		]
	}),
	'grpc.server_max_unrequested_time_in_server': 7 * 60,  # 7 minutes.
}

# TODO: make this non-protected.
# noinspection PyProtectedMember
GrpcCommunicator._default_grpc_options |= {
	'grpc.service_config': json.dumps({
		'methodConfig': [
			{
				'name': [{}],
				'retryPolicy': {
					'maxAttempts': 5,
					'initialBackoff': '1s',
					'maxBackoff': '7s',
					'backoffMultiplier': 2,
					'retryableStatusCodes': ['UNAVAILABLE'],
				},
			}
		]
	}),
	'grpc.server_max_unrequested_time_in_server': 7 * 60,  # 7 minutes.
}

if conf.TORCH_INTRA_OP_THREADS is not None:
	torch.set_num_threads(conf.TORCH_INTRA_OP_THREADS)
if conf.TORCH_INTER_OP_THREADS is not None:
	torch.set_num_interop_threads(conf.TORCH_INTER_OP_THREADS)
logging.debug(f"Torch intra-op threads: {torch.get_num_threads()}; inter-op threads: {torch.get_num_interop_threads()}")

torch_context = TorchContext(
	model=model,
	optimizer=optim.init(),
	criterion=CrossEntropyLoss(),  # TODO: move into a module to share with the `finetunes.lr` and the `finetunes.prune` notebooks.
	train_data_loader=train_data_loader,
	train_eval_data_loader=train_eval_data_loader,
	test_data_loader=test_data_loader,
	device=conf_eval.DEVICE_EVAL,
)
logging.debug(f"Using device {conf.DEVICE}, and device {conf.AGGREGATOR_DEVICE} for aggregations.")

logging.info(f"Initial peers: {conf_eval.INITIAL_PEERS_EVAL}.", extra={'type': 'initial-peers', 'peers': list(conf_eval.INITIAL_PEERS_EVAL)})

optim_lr_scheduler: ExponentialLR | None = None


def _reset_optim_lr_scheduler() -> None:
	global optim_lr_scheduler
	
	optim_lr_scheduler = ExponentialLR(torch_context.optimizer, gamma=conf.LR_DECAY)
	
	# TODO: any better way than this loop?
	for _ in range(singletons.round_ + 1):
		optim_lr_scheduler.step()
	
	logging.debug(f"Reset the optimizer's LR scheduler; current LR = {optim_lr_scheduler.get_last_lr()}.")


if conf.LR_DECAY is not None:
	_reset_optim_lr_scheduler()


async def _reset_optim() -> None:
	# TODO: only reset if the model parameters are affected by the aggregation; i.e., do not reset if aggregated with no one.
	
	torch_context.optimizer = optim.init()
	
	if optim_lr_scheduler is not None:
		_reset_optim_lr_scheduler()
	
	logging.debug("Reset the optimizer.")


def param_agg_factory(**kwargs):
	return TensorToDeviceDecoratingAggregator(conf_eval.AGGREGATOR_DEVICE_EVAL, WeightedAvgTensorsAggregator(weight_meta_key='data_len'))


async def _post_train() -> None:
	if conf.MONITOR_SIMILARITIES != 'none':
		await search_data.publish()
	
	if optim_lr_scheduler is not None:
		optim_lr_scheduler.step()


async def _init_grpc_server() -> None:
	# noinspection PyProtectedMember
	singletons.grpc_server = grpc.aio.server(options=list((P2PSimsNode.grpc_options | GrpcCommunicator._default_grpc_options).items()))
	singletons.grpc_server.add_insecure_port('[::]:50051')


async def _cfl_server_communicator_receive_handler(meta: Meta | None) -> Any:
	match meta['from']:
		case 'server':
			return await singletons.cfl_client.receive_handler(meta)
		case _:
			return await singletons.cfl_server.receive_handler(meta)


async def _init_dfl_communicator() -> None:
	singletons.base_dfl_communicator = await GrpcCommunicator(name=conf.NAME, init_server=singletons.grpc_server, encoder=ChainEncoder((TensorBytesEncoder(load_weights_only=True), PickleBytesEncoder(),)))
	singletons.dfl_communicator = LoggingCommunicatorDecorator(singletons.base_dfl_communicator, logger=logging.getLogger('dfl.communication.grpc'))
	singletons.dfl_caching_communicator = await StrongSubscribeCachingCommunicatorDecorator(singletons.dfl_communicator, logger=logging.getLogger('dfl.communication.caching'))


async def _init_dfl_node() -> None:
	singletons.dfl_node = await DflNode(
		torch_context=torch_context,
		communicator=singletons.dfl_communicator,
		select_neighbors=peers_selection.do,
		param_agg_factory=param_agg_factory,
		cleanup_rounds_age=conf.ROUNDS_AGE,
		extra_publish_meta={'data_len': len(train_data)},
		pull_early=conf.PULL_EARLY_PEERS_SELECTION,
		post_train=lambda: _post_train(),
	)
	
	if not conf.DISABLE_CDFL and conf.INDEX == 0:
		singletons.cdfl_node = await CDflNode(logger=logging.getLogger('cdfl'))
		singletons.base_dfl_communicator._receive_handler = singletons.cdfl_node.receive_handler


async def _init_cfl_server_client() -> None:
	if conf.INDEX == 0:
		singletons.cfl_server = await CflServer(
			communicator=singletons.dfl_communicator,
			agg_factory=cfl.agg_factory,
			agg_type='clustered' if conf.CENTRALIZED_AGGREGATOR == 'clustering' else 'selective' if conf.CENTRALIZED_AGGREGATOR.startswith('ideal-top-') else 'all',
			clients=conf.ALL_NODES,
			logger=logging.getLogger('cfl_server'),
			name='server'
		)
	
	singletons.cfl_client = await CflClient(
		torch_context=torch_context,
		communicator=singletons.dfl_communicator,
		server='server',
		logger=logging.getLogger('cfl_client'),
		extra_publish_meta={'data_len': len(train_data)},
		post_train=lambda: _post_train()
	)
	singletons.dfl_node = singletons.cfl_client
	
	singletons.base_dfl_communicator._receive_handler = _cfl_server_communicator_receive_handler if conf.INDEX == 0 else singletons.cfl_client.receive_handler


async def _init_asyncio_io_threads() -> None:
	if conf.ASYNCIO_IO_THREADS is not None:
		asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(thread_name_prefix='asyncio', max_workers=conf.ASYNCIO_IO_THREADS))


async def _train_test_eval():
	if not conf.CENTRALIZED:
		node = singletons.dfl_node
	else:
		node = singletons.cfl_client
	
	logging.debug(f"Train/Test evaluating the model (round #{node.round_})...")
	
	if conf.MODEL == 'cnn':  # Due to a memory spike if doing both at the same time; only happens in the CNN model. TODO: why though?
		await node.train_eval()
		await node.test_eval()
	else:
		async with asyncio.TaskGroup() as tg:
			tg.create_task(node.train_eval())
			tg.create_task(node.test_eval())
	
	logging.debug(f"Train/Test evaluated the model (round #{node.round_}).")


async def _gc():
	await asyncio.to_thread(gc.collect)
	await asyncio.to_thread(torch.cuda.empty_cache)
	
	logging.debug("GC-ed")


async def _close() -> None:
	if singletons.cdfl_node is not None:
		await singletons.cdfl_node.cleanup(all_rounds=True)
		singletons.cdfl_node = None
	
	if singletons.cfl_client is not None:
		await singletons.cfl_client.cleanup(all_rounds=True)
		singletons.cfl_client = None
		
		if singletons.dfl_node == singletons.cfl_client:
			singletons.dfl_node = None
	
	if singletons.dfl_node is not None:
		await singletons.dfl_node.cleanup(all_rounds=True)
		singletons.dfl_node = None
	
	if singletons.cfl_server is not None:
		await singletons.cfl_server.cleanup(all_rounds=True)
		singletons.cfl_server = None
	
	if singletons.dfl_caching_communicator is not None:
		await singletons.dfl_caching_communicator.close(close_delegate=False)
		singletons.dfl_caching_communicator = None
	
	if singletons.base_dfl_communicator is not None:
		await singletons.base_dfl_communicator.close(grace=timedelta(seconds=7))
		singletons.base_dfl_communicator = None
		singletons.dfl_communicator = None
	
	await P2PSimsNode.release_all_stubs()
	
	if singletons.grpc_server is not None:
		await singletons.grpc_server.stop(grace=7)
		singletons.grpc_server = None
	
	if sync.sync_manager is not None:
		if conf.INDEX == 0:
			await asyncio.to_thread(sync.sync_manager.shutdown)
		sync.sync_manager = None


async def amain_cfl():
	await _init_asyncio_io_threads()
	await conf_eval.SIMULATE_LATENCIES_EVAL()
	
	try:
		await _init_grpc_server()
		await _init_dfl_communicator()
		
		await singletons.grpc_server.start()
		if conf.SYNC_INIT_GRPC_SERVER:
			await sync.sync('gRPC server init.')  # Ensure that all nodes' gRPC servers are accessible.
		
		await _init_cfl_server_client()
		
		if conf.INDEX == 0:
			singletons.cfl_server.round_ = -1
		singletons.cfl_client.round_ = -1
		
		if conf.PRENUP_ROUND:
			async with asyncio.TaskGroup() as tg:
				tg.create_task(_train_test_eval())
				
				if not conf.GRADS_BASED_SEARCH_DATA:
					tg.create_task(search_data.publish())
					tg.create_task(allsims.step())
				else:
					allsims.step_state()
			
			if conf.GC_PER_ROUND:
				await _gc()
			
			if conf.SYNC_ROUNDS:
				await sync.sync_round()
		
		if conf.INDEX == 0:
			singletons.cfl_server.round_ = 0
		singletons.cfl_client.round_ = 0
		singletons.round_ = 0
		
		for _ in range(conf.ROUNDS):
			async with asyncio.TaskGroup() as tg:
				if conf.INDEX == 0:
					tg.create_task(singletons.cfl_server.step_round())
				tg.create_task(singletons.cfl_client.step_round(epochs=conf.FIRST_ROUND_TRAIN_EPOCHS if singletons.round_ == 0 else conf.EPOCHS))
				tg.create_task(allsims.step())
			
			if conf.OPTIMIZER_RESET:
				await _reset_optim()
			
			# FIXME: reset the search data here?
			
			if conf.GC_PER_ROUND:
				await _gc()
			
			singletons.round_ += 1
			
			if conf.SYNC_ROUNDS:
				await sync.sync_round()
		
		if conf.SYNC_EXIT:
			await sync.sync_exit()
	finally:
		await _close()


async def amain_dfl():
	await _init_asyncio_io_threads()
	await conf_eval.SIMULATE_LATENCIES_EVAL()
	
	try:
		await _init_grpc_server()
		await _init_dfl_communicator()
		p2psims.node.add_servicer_to_server(singletons.p2psims_sessions_nodes, singletons.grpc_server)
		
		await singletons.grpc_server.start()
		if conf.SYNC_INIT_GRPC_SERVER:
			await sync.sync('gRPC server init.')  # Ensure that all nodes' gRPC servers are accessible.
		
		await _init_dfl_node()
		singletons.dfl_node.round_ = -1
		if not conf.DISABLE_CDFL and conf.INDEX == 0:
			singletons.cdfl_node.round_ = -1
		
		if conf.PRENUP_ROUND:
			async with asyncio.TaskGroup() as tg:
				tg.create_task(_train_test_eval())
				
				if not conf.GRADS_BASED_SEARCH_DATA:
					tg.create_task(search_data.publish())
					tg.create_task(allsims.step())
				else:
					allsims.step_state()
			
			if conf.GC_PER_ROUND:
				await _gc()
			
			if conf.SYNC_ROUNDS:
				await sync.sync_round()
		else:
			allsims.step_state()
		
		singletons.dfl_node.round_ = 0
		if not conf.DISABLE_CDFL and conf.INDEX == 0:
			singletons.cdfl_node.round_ = 0
		singletons.round_ = 0
		
		for _ in range(conf.ROUNDS):
			if conf.SYNC_ROUNDS and conf.ROUNDS_AGE == 0:
				async with asyncio.TaskGroup() as tg:
					tg.create_task(singletons.dfl_node.step_round(epochs=conf.FIRST_ROUND_TRAIN_EPOCHS if singletons.round_ == 0 else conf.EPOCHS, cleanup=False))
					if not conf.DISABLE_CDFL and conf.INDEX == 0:
						tg.create_task(singletons.cdfl_node.step_round(cleanup=False))
					tg.create_task(allsims.step())
			else:
				async with asyncio.TaskGroup() as tg:
					tg.create_task(singletons.dfl_node.step_round(epochs=conf.FIRST_ROUND_TRAIN_EPOCHS if singletons.round_ == 0 else conf.EPOCHS))
					if not conf.DISABLE_CDFL and conf.INDEX == 0:
						tg.create_task(singletons.cdfl_node.step_round())
					tg.create_task(allsims.step())
			
			if conf.OPTIMIZER_RESET:
				await _reset_optim()
			
			await search_data.reset()
			
			if conf.GC_PER_ROUND:
				await _gc()
			
			singletons.round_ += 1
			
			if conf.SYNC_ROUNDS:
				await sync.sync_round()
				
				if conf.ROUNDS_AGE == 0:
					if not conf.DISABLE_CDFL and conf.INDEX == 0:
						await singletons.cdfl_node.cleanup()
					await singletons.dfl_node.cleanup()
					
					if conf.GC_PER_ROUND:
						await _gc()
		if conf.SYNC_EXIT:
			await sync.sync_exit()
	finally:
		await _close()


asyncio.run(amain_dfl() if not conf.CENTRALIZED else amain_cfl())
