import asyncio
import logging
import multiprocessing
import socket
from multiprocessing.managers import SyncManager

from tenacity import wait_random_exponential, retry, before_sleep_log, retry_if_exception_type

import conf
from exts.asyncio_exts import once

# TODO: migrate to a communicator-based async solution rather than this thread-per-connection-based solution.

_logger = logging.getLogger(__name__)


class MyManager(SyncManager):
	pass


sync_manager: MyManager | None = None


@once
async def init() -> None:
	global sync_manager
	
	if conf.INDEX == 0:
		barrier = multiprocessing.Barrier(parties=conf.NODES_COUNT)
		MyManager.register('barrier', lambda: barrier)
		
		# noinspection PyTypeChecker
		sync_manager = MyManager(address=('0.0.0.0', 5000), authkey=b'', shutdown_timeout=None)
		sync_manager.start()
		
		_logger.debug("Started the sync manager server.")
	else:
		MyManager.register('barrier')
		
		@retry(wait=wait_random_exponential(min=1, max=7), before_sleep=before_sleep_log(_logger, log_level=logging.DEBUG), retry=retry_if_exception_type(socket.gaierror))
		async def _get_n0_ip():
			return await asyncio.to_thread(socket.gethostbyname, 'n0')
		
		# noinspection PyTypeChecker
		sync_manager = MyManager(address=(await _get_n0_ip(), 5000), authkey=b'', shutdown_timeout=None)
		
		@retry(wait=wait_random_exponential(min=1, max=7), before_sleep=before_sleep_log(_logger, log_level=logging.DEBUG), retry=retry_if_exception_type(ConnectionRefusedError))
		async def _connect():
			await asyncio.to_thread(sync_manager.connect)
			_logger.debug("The sync manager connected.")
		
		await _connect()


# This failed in some runs with a transient `No route to host` error; therefore retries.
@retry(wait=wait_random_exponential(min=1, max=7), before_sleep=before_sleep_log(_logger, log_level=logging.DEBUG), retry=retry_if_exception_type(OSError))
async def _get_barrier():
	# noinspection PyUnresolvedReferences
	return await asyncio.to_thread(sync_manager.barrier)


async def sync_exit() -> None:
	# FIXME: ensure the server (`n0`) is the last one exiting this function; otherwise, occasional, probably non-harmful, `EOFError` happens.
	try:
		await sync('exit')
	except EOFError as e:
		# All parties waited, but n0 itself acted and exited sooner than me receiving a reply.
		_logger.warning("The sync manager server node (probably `n0`) acted and exited sooner than me receiving a reply; this isn't generally harmful.", exc_info=e)


async def sync_round() -> None:
	await sync('round')


async def sync(cause: str | None = None) -> None:
	await init()
	
	barrier = await _get_barrier()
	
	if cause is None:
		_logger.debug("Waiting on the global sync...")
	else:
		_logger.debug(f"Waiting on the global sync ({cause})...")
	
	await asyncio.to_thread(barrier.wait)
	
	if cause is None:
		_logger.debug("Synced globally.")
	else:
		_logger.debug(f"Synced globally ({cause}).")
