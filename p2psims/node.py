import asyncio
import ipaddress
import logging
import math
import pickle
import platform
import random
import socket
import sys
import traceback
from datetime import timedelta, datetime, timezone
from functools import cached_property
from ipaddress import IPv4Address
from typing import Any, Iterable, override, Callable, Awaitable, Union, Literal, Hashable

import binpacking
import sympy
import tenacity.wait
from google.protobuf.empty_pb2 import Empty
from grpc.aio import insecure_channel, Channel, server as grpc_server, Server as GrpcServer, ServicerContext

import p2psims.extensions.binpacking as binpacking_exts
import p2psims.extensions.logging as logging_exts
from p2psims.aggregation import np_mean
from p2psims.clustering import affinity_propagation
from p2psims.extensions.datetime import EPOCH
from p2psims.protos.initiator_pb2 import JoinZoneRequest, JoinZoneDecision
from p2psims.protos.initiator_pb2_grpc import ZoneInitiatorServicer, ZoneInitiatorStub, add_ZoneInitiatorServicer_to_server
from p2psims.protos.peer_pb2 import SuggestJoinZoneRequest, SuggestClusterRepresentativeRequest, SuggestClusterRepresentativeResponse, Data, MigrateZoneInitiatorRequest, PromoteToZoneInitiatorRequest
from p2psims.protos.peer_pb2_grpc import PeerServicer, PeerStub, add_PeerServicer_to_server
from p2psims.protos.representative_pb2 import PromoteToSuperClusterRepresentativeRequest, NodeIdentity, SearchRequest, SearchResponse, ImmediateSearchRequest, ImmediateSearchResponse
from p2psims.protos.representative_pb2_grpc import ClusterRepresentativeServicer, ClusterRepresentativeStub, add_ClusterRepresentativeServicer_to_server
from p2psims.similarity import cosine

_logger = logging.getLogger('p2psims.node')


class Servicer(PeerServicer, ZoneInitiatorServicer, ClusterRepresentativeServicer):
	pass


def add_servicer_to_server(servicer: Servicer, server: GrpcServer) -> None:
	add_PeerServicer_to_server(servicer, server)
	add_ZoneInitiatorServicer_to_server(servicer, server)
	add_ClusterRepresentativeServicer_to_server(servicer, server)


class Stub(PeerStub, ZoneInitiatorStub, ClusterRepresentativeStub):
	def __init__(self, channel: Channel):
		PeerStub.__init__(self, channel)
		ZoneInitiatorStub.__init__(self, channel)
		ClusterRepresentativeStub.__init__(self, channel)


class IpTimeModuloBeInitiator:
	# The default 1 microsecond time deviation is the most extreme value, effectively disabling its effect in the initiator selection process.
	def __init__(self, advertise_ip: ipaddress.IPv4Address, initial_modulo: int, time_deviation: timedelta = timedelta(microseconds=1), logger: logging.Logger | logging.LoggerAdapter | None = None) -> None:
		self._advertise_ip = advertise_ip
		self._initial_modulo = initial_modulo
		
		self._time_deviation_microseconds = (time_deviation.days * 86400 + time_deviation.seconds) * 10 ** 6 + time_deviation.microseconds
		
		if logger is None:
			logger = _logger
		self._logger = logger
		
		self._modulo_factors = sorted(sympy.factorint(self._initial_modulo, multiple=True))
		
		# noinspection PyTypeChecker
		if self._time_deviation_microseconds % initial_modulo == 0:
			self._logger.warning("time_deviation % initial_modulo is zero, therefore the IP alone determines being an initiator or not; as the IP is a constant, the calls would always return True or False, and that might not be an expected behavior.")
	
	async def __call__(self, attempt: int, *args, **kwargs) -> bool:
		now_epoch = datetime.now(timezone.utc) - EPOCH
		now_epoch_microseconds = (now_epoch.days * 86400 + now_epoch.seconds) * 10 ** 6 + now_epoch.microseconds
		
		assert isinstance(self._time_deviation_microseconds, int)
		assert isinstance(now_epoch_microseconds, int)
		
		t = round(now_epoch_microseconds / self._time_deviation_microseconds) * self._time_deviation_microseconds
		r = int(self._advertise_ip) + t
		
		modulo = self._initial_modulo
		for factor in self._modulo_factors[:attempt - 1]:
			modulo //= factor
		
		self._logger.debug(f"The modulo for attempt {attempt} is {modulo}.")
		
		return r % modulo == 0


class Node(Servicer):
	def __init__(
			self,
			data: Any | dict[str, Any],
			initial_peers_addresses: Iterable[str],
			preferred_zone_size: int,  # TODO: determine a default value?
			no_probe_duration: timedelta | tenacity.wait.wait_base,  # TODO: determine a default value?
			level: int = 0,
			init_grpc_server: bool | GrpcServer = True,
			grpc_server_listen_address: str = "[::]:50051",
			advertise_address: str | None = None,
			advertise_ip: ipaddress.IPv4Address | None = None,
			cluster_data: Callable[[dict[str, Any]], Awaitable[dict[str, set[str]]]] = affinity_propagation,
			aggregate_data: Callable[[Iterable[Any]], Awaitable[Any]] = np_mean,
			be_initiator: bool | Callable[[int], Awaitable[bool]] = True,
			similarity: Callable[[Any, Any], Awaitable[float]] = cosine,
			zone_partition_method: Literal['data', 'peers', 'zones-data'] = 'peers',
			logger: logging.Logger | logging.LoggerAdapter | None = None,
			session: str | None = None,
			max_zone_size: int | None = None,
			level_multiplex_servicer: Union['LevelMultiplexingServicer', None] = None
	):
		self._peers_addresses = set(initial_peers_addresses)
		self._preferred_zone_size = preferred_zone_size
		self._level = level
		self._init_grpc_server = init_grpc_server
		self._grpc_server_listen_address = grpc_server_listen_address
		self._max_zone_size = max_zone_size
		
		if isinstance(no_probe_duration, timedelta):
			no_probe_duration = tenacity.wait.wait_fixed(no_probe_duration)
		self._no_probe_duration = no_probe_duration
		
		if advertise_address is None:
			# TODO: log the inferred advertise address.
			advertise_address = f"{platform.node()}:{self._grpc_server_listen_address.rsplit(':')[-1]}"
		self._advertise_address = advertise_address
		
		if not isinstance(data, dict):  # TODO: what if the actual datum (a single data) is a dict?
			data = {self._advertise_address: data}
		self._data = data
		
		self._advertise_ip = advertise_ip
		
		if logger is None:
			logger = _logger
		logger = logging_exts.LoggerAdapter(logger, extra={'level': self._level}, overwrite=False)
		self._logger = logger
		
		self._cluster_data = cluster_data
		self._aggregate_data = aggregate_data
		self._be_initiator = be_initiator
		self._similarity = similarity
		self._zone_partition_method = zone_partition_method
		self.session = session
		self._level_multiplex_servicer = level_multiplex_servicer
		
		self._is_cluster_representative = False
		if self._level == 0:
			self._search_cluster_representative_event = asyncio.Event()
		
		self._tasks: list[asyncio.Task] = []
		
		self._informed_initiators = set[str]()
		self._informed_initiators_lock = asyncio.Lock()
		
		self._suggestee_peers = set[str]()
		self._suggestee_peers_lock = asyncio.Lock()
		
		self._initiator_address: str | None = None
		
		self._logger.debug("Initialized.")
	
	async def __ainit__(self):
		if self._advertise_ip is None:
			self._advertise_ip = IPv4Address(await asyncio.to_thread(socket.gethostbyname, self._advertise_address.rsplit(':', maxsplit=1)[0]))
			self._logger.debug(f"Inferred the advertise IP: {self._advertise_ip}.")
		
		if self._be_initiator is None or self._be_initiator == True:
			self._be_initiator = IpTimeModuloBeInitiator(self._advertise_ip, initial_modulo=self._preferred_zone_size, logger=self._logger)
		
		self._is_initiator_v = await self._is_initiator()
		if self._is_initiator_v:
			self._logger.info("Selected to be a zone initiator.", extra={'type': 'zone-initiator'})
			
			await self._init_initiator()
		else:
			self._join_zone_lock = asyncio.Lock()
			
			if self._be_initiator:
				self._no_probe_watchdog_task = asyncio.create_task(self._no_probe_watchdog())
				self._logger.debug("Created the no-probe watchdog.")
			
			self._logger.debug("Waiting for a join zone suggestion...")  # FIXME: this should not be logged in case of a representative-only next-level node.
		
		if isinstance(self._init_grpc_server, GrpcServer):
			assert self._level == 0
			
			if self._level_multiplex_servicer is None:
				self._level_multiplex_servicer = LevelMultiplexingServicer()
			
			add_servicer_to_server(self._level_multiplex_servicer, self._init_grpc_server)
		elif self._init_grpc_server:
			assert self._level == 0
			
			if self._level_multiplex_servicer is None:
				self._level_multiplex_servicer = LevelMultiplexingServicer()
			
			self._logger.debug("Initializing the gRPC server...")
			
			self._grpc_server = grpc_server(options=list(Node.grpc_options.items()))
			self._grpc_server.add_insecure_port(self._grpc_server_listen_address)
			add_servicer_to_server(self._level_multiplex_servicer, self._grpc_server)
			
			await self._grpc_server.start()
			
			self._logger.debug("Initialized the gRPC server.")
		else:
			assert self._level_multiplex_servicer is not None
			pass  # NOP.
		
		if self._is_initiator_v:
			self._tasks.append(asyncio.create_task(self._become_initiator()))
		
		await self._level_multiplex_servicer.register_level_servicer(self._level, self)
	
	@cached_property
	def _flattened_zone_data(self) -> dict[str, Any]:
		# TODO: optimize this property to be a read-only view on the flattened `self._zone_data`.
		
		r = {}
		for d in self._zone_data.values():
			r.update(d)
		return r
	
	def _invalidate_flattened_zone_data_cache(self) -> None:
		try:
			# noinspection PyPropertyAccess
			del self._flattened_zone_data
		except AttributeError:
			pass
	
	async def _become_initiator(self, skip_suggesting: bool = False, skip_partitioning: bool = False) -> None:
		if not skip_suggesting:
			self._logger.debug("Suggesting peers to join my zone...")
			async with asyncio.TaskGroup() as tg:
				async with self._suggestee_peers_lock:
					peers = self._peers_addresses - self._suggestee_peers
				
				for peer_address in peers:
					tg.create_task(self._suggest_initiator(peer_address))
			self._logger.debug("Finished suggesting peers to join my zone.")
		
		if not skip_partitioning and (self._max_zone_size is not None and len(self._flattened_zone_data if self._zone_partition_method == 'data' else self._zone_data) > self._max_zone_size):
			self._logger.debug(f"Too big of a zone {dict((peer, set(data.keys())) for peer, data in self._zone_data.items())}; will partition...")
			await self._partition_zone()
		
		self._logger.info(f"Zone peers: {set(self._zone_data.keys())}.", extra={'type': 'zone', 'peers': list(self._zone_data.keys())})
		
		self._logger.debug("Clustering zone data...")
		self._zone_clusters = await self._cluster_data(self._flattened_zone_data)
		self._logger.debug(f"Clustered zone data: {self._zone_clusters}.")
		
		self._logger.debug("Assigning zone clusters' representatives...")
		await self._assign_zone_clusters_representatives()
		self._logger.debug("Assigned zone clusters' representatives.")
		
		# Release the no longer needed raw zone data.
		del self._zone_data
		del self._zone_data_lock
		self._invalidate_flattened_zone_data_cache()
		
		await self._proceed_next_level()
		
		# Release the no longer needed zone clusters' data.
		del self._zone_clusters_data
	
	async def _is_initiator(self, attempt: int = 1) -> bool:
		# This function is kept for backward compatiblity with the old bool-typed `self._be_initiator`.
		
		if self._be_initiator == False:
			return False
		
		return await self._be_initiator(attempt=attempt)
	
	async def _init_initiator(self):
		self._initiator_address = self._advertise_address
		
		self._zone_data: dict[str, dict[str, Any]] = {self._advertise_address: self._data}
		self._zone_data_lock = asyncio.Lock()
		
		self._neighbor_initiators_addresses: set[str] = set()
	
	async def _no_probe_watchdog(self) -> None:
		# TODO: to be the most accurate, should initially wait for the gRPC server to start (if `self._init_grpc_server`).
		
		# TODO: properly implement this using a retrying function.
		# noinspection PyTypeChecker
		retry_state = tenacity.RetryCallState(None, None, None, None)
		while True:
			wait = self._no_probe_duration(retry_state)
			
			self._logger.debug(f"No-probe watchdog sleeping for {wait} seconds...")
			await asyncio.sleep(wait)
			self._logger.debug("No-probe watchdog woke up.")
			
			retry_state.prepare_for_next_attempt()
			
			if self._initiator_address is None:
				async with self._join_zone_lock:
					if self._initiator_address is None:
						self._logger.debug("No-probe mechanism triggered...")
						
						is_initiator_v = await self._is_initiator(attempt=retry_state.attempt_number)
						self._logger.debug(f"Will become an initiator? {'Yes' if is_initiator_v else 'No'}.")
						
						if is_initiator_v:
							self._logger.info("Selected to be a zone initiator (by the no-probe watchdog).", extra={'type': 'zone-initiator'})
							
							await self._init_initiator()
							self._is_initiator_v = True
						else:
							pass  # NOP; proceed to the next iteration.
					else:
						break  # Already joined a zone.
				
				if self._is_initiator_v:
					await self._become_initiator()
					break
			else:
				break  # Already joined a zone.
		
		self._logger.debug("No-probe watchdog exiting...")
	
	async def _partition_zone(self) -> None:
		match self._zone_partition_method:
			case 'data':
				assert all(len(data) <= self._preferred_zone_size for data in self._zone_data.values())
				
				zones_peers_addresses = binpacking.to_constant_volume(
					d=list(self._zone_data.keys()),
					key=lambda peer_address: len(self._zone_data[peer_address]),
					V_max=self._preferred_zone_size
				)
			case 'peers':
				zones_peers_addresses = binpacking_exts.to_constant_bin_number_with_max_items_per_bin_and_fair_volume(
					items=list(self._zone_data.keys()),
					bins_count=math.ceil(len(self._zone_data) / self._preferred_zone_size),
					max_items_per_bin=self._preferred_zone_size,
					weight_key=lambda peer_address: len(self._zone_data[peer_address])
				)
			case 'zones-data':
				# Try to distribute as fairly as possible based on their weights.
				zones_peers_addresses = binpacking.to_constant_bin_number(
					d=list(self._zone_data.keys()),
					key=lambda peer_address: len(self._zone_data[peer_address]),
					N_bin=math.ceil(len(self._zone_data) / self._preferred_zone_size)
				)
			case _:
				raise ValueError(f"Unknown zone partition method \"{self._zone_partition_method}\".")
		
		zones_data = [{peer_address: self._zone_data[peer_address] for peer_address in zone_peers_addresses} for zone_peers_addresses in zones_peers_addresses]
		
		async with asyncio.TaskGroup() as tg:
			for zone_data in zones_data:
				if self._advertise_address in zone_data:
					self._zone_data = zone_data
					self._invalidate_flattened_zone_data_cache()
				else:
					tg.create_task(self._promote_to_zone_initiator(zone_data))
		
		self._logger.debug(f"Partitioned the zone into the followings: {[dict((peer, set(data.keys())) for peer, data in zone_data.items()) for zone_data in zones_data]}.")
	
	async def _promote_to_zone_initiator(self, zone_data: dict[str, dict[str, Any]], zone_initiator_address: str | None = None) -> None:
		if zone_initiator_address is None:
			zone_initiator_address = next(iter(zone_data.keys()))
		
		# noinspection PyShadowingNames
		async def _migrate_zone_initiator(peer_address: str) -> None:
			peer_stub = await Node._get_stub(peer_address)
			await peer_stub.MigrateZoneInitiator(MigrateZoneInitiatorRequest(new_zone_initiator_address=zone_initiator_address), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
		
		async with asyncio.TaskGroup() as tg:
			for peer_address in set(zone_data.keys()) - {zone_initiator_address}:
				tg.create_task(_migrate_zone_initiator(peer_address))
		
		zone_initiator_stub = await Node._get_stub(zone_initiator_address)
		await zone_initiator_stub.PromoteToZoneInitiator(PromoteToZoneInitiatorRequest(peers_addresses_data={peer_address: Data(peers_addresses_data={peer_address: pickle.dumps(datum) for peer_address, datum in data.items()}) for peer_address, data in zone_data.items() if peer_address != zone_initiator_address}), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
		self._neighbor_initiators_addresses.add(zone_initiator_address)
	
	async def _proceed_next_level(self) -> None:
		if len(self._neighbor_initiators_addresses) == 0:
			self._logger.debug("Reached the super initiator.")
			
			self._logger.debug("Promoting the zone clusters' representatives to super cluster representatives...")
			async with asyncio.TaskGroup() as tg:
				for cluster_representative_address in self._zone_clusters.keys():
					tg.create_task(self._promote_to_super_cluster_representative(cluster_representative_address))
			self._logger.debug("Promoted the zone clusters' representatives to super cluster representatives.")
			
			return
		
		next_level = self._level + 1
		
		self._logger.debug(f"Proceeding to the next level #{next_level}...")
		
		self._logger.debug("Initializing the next level node...")
		next_level_node = await Node(
			data=self._zone_clusters_data,
			initial_peers_addresses=self._neighbor_initiators_addresses,
			preferred_zone_size=self._preferred_zone_size,
			no_probe_duration=self._no_probe_duration,
			level=next_level,
			init_grpc_server=False,
			grpc_server_listen_address=self._grpc_server_listen_address,
			advertise_address=self._advertise_address,
			advertise_ip=self._advertise_ip,
			cluster_data=self._cluster_data,
			aggregate_data=self._aggregate_data,
			be_initiator=True if self._be_initiator == False else self._be_initiator,  # `True` in case of `False` is a logic to keep the backward compatibility before callable `self._be_initiator`. TODO: is this an expected behavior?
			similarity=self._similarity,
			logger=self._logger,
			session=self.session,
			max_zone_size=self._max_zone_size,
			level_multiplex_servicer=self._level_multiplex_servicer
		)
		self._logger.debug(f"Initialized the next level #{next_level} node.")
		
		self._logger.debug(f"Proceeded to the next level #{next_level}.")
	
	async def _proceed_representative_next_level(self) -> None:
		next_level = self._level + 1
		
		self._logger.debug(f"Proceeding to the next level #{next_level} (for representative only)...")
		
		self._logger.debug(f"Initializing the next level #{next_level} node (for representative only)...")
		# TODO: there are unused stuff in this kind of node; optimize that.
		next_level_node = await Node(
			data={},
			initial_peers_addresses=self._peers_addresses,
			preferred_zone_size=self._preferred_zone_size,
			no_probe_duration=self._no_probe_duration,
			level=next_level,
			init_grpc_server=False,
			grpc_server_listen_address=self._grpc_server_listen_address,
			advertise_address=self._advertise_address,
			advertise_ip=self._advertise_ip,
			cluster_data=self._cluster_data,
			aggregate_data=self._aggregate_data,
			be_initiator=False,
			similarity=self._similarity,
			logger=self._logger,
			session=self.session,
			max_zone_size=self._max_zone_size,
			level_multiplex_servicer=self._level_multiplex_servicer
		)
		self._logger.debug(f"Initialized the next level #{next_level} node (for representative only).")
		
		self._logger.debug(f"Proceeded to the next level #{next_level} (for representative only).")
	
	async def _promote_to_super_cluster_representative(self, cluster_representative_address: str) -> None:
		representative_stub = await Node._get_stub(cluster_representative_address)
		await representative_stub.PromoteToSuperClusterRepresentative(PromoteToSuperClusterRepresentativeRequest(super_clusters_representatives_addresses=tuple(k for k in self._zone_clusters_data.keys() if k != cluster_representative_address)), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
	
	async def _assign_zone_clusters_representatives(self) -> None:
		self._zone_clusters_data = dict[str, Any]()
		new_zone_clusters = {}
		
		# noinspection PyShadowingNames
		async def _coro(cluster_key: str):
			cluster_peers_addresses = self._zone_clusters[cluster_key]
			cluster_datum = await self._aggregate_data([self._flattened_zone_data[peer] for peer in cluster_peers_addresses])
			
			# Force a random peer.
			
			peer_address = random.choice(sorted(cluster_peers_addresses))
			
			self._logger.debug(f"Forcing {peer_address} to be the cluster representative of {cluster_key}...")
			
			peer_stub = await Node._get_stub(peer_address)
			res = await peer_stub.SuggestClusterRepresentative(SuggestClusterRepresentativeRequest(force=True, data=pickle.dumps(cluster_datum), peers_addresses=cluster_peers_addresses), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
			if not res.accept:
				raise NotImplementedError
			
			self._logger.debug(f"Peer {peer_address} accepted the forced cluster representative burden.")
			
			return peer_address, cluster_datum, cluster_peers_addresses
		
		tasks = []
		async with asyncio.TaskGroup() as tg:
			for cluster_key in self._zone_clusters.keys():
				tasks.append(tg.create_task(_coro(cluster_key)))
			
			# TODO: migrate to `async for` after updating to Python 3.13.
			for task in asyncio.as_completed(tasks):
				cluster_representative_address, cluster_datum, cluster_peers_addresses = await task
				
				new_zone_clusters[cluster_representative_address] = cluster_peers_addresses
				self._zone_clusters_data[cluster_representative_address] = cluster_datum
		
		self._zone_clusters = new_zone_clusters
	
	async def _reject_initiator(self, initiator_address: str) -> None:
		# TODO: take a local route in case I'm the initiator myself.
		suggested_initiator_stub = await Node._get_stub(initiator_address)
		await suggested_initiator_stub.JoinZone(JoinZoneRequest(decision=JoinZoneDecision.REJECT, initiator_address=self._initiator_address), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
		
		self._logger.debug(f"Rejected zone initiator `{initiator_address}`.")
	
	async def _inform_self_initiator(self, neighbor_initiator_address: str) -> None:
		# TODO: take a local route in case I'm the initiator myself.
		self_initiator_stub = await Node._get_stub(self._initiator_address)
		await self_initiator_stub.JoinZone(JoinZoneRequest(decision=JoinZoneDecision.INFORM, initiator_address=neighbor_initiator_address), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
		
		self._logger.debug(f"Informed self initiator of `{neighbor_initiator_address}`.")
	
	async def _accept_initiator(self) -> None:
		self_initiator_stub = await Node._get_stub(self._initiator_address)
		await self_initiator_stub.JoinZone(JoinZoneRequest(decision=JoinZoneDecision.ACCEPT, data=Data(peers_addresses_data={peer_address: pickle.dumps(datum) for peer_address, datum in self._data.items()})), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
		
		self._logger.info(f"Accepted zone initiator `{self._initiator_address}`.", extra={'type': 'zone-join', 'initiator': self._initiator_address})
	
	async def _suggest_initiator(self, peer_address: str):
		async with self._suggestee_peers_lock:
			if peer_address in self._suggestee_peers:
				return
		
		self._logger.debug(f"Suggesting `{peer_address}` to join my zone...")
		
		peer_stub = await Node._get_stub(peer_address)
		await peer_stub.SuggestJoinZone(SuggestJoinZoneRequest(initiator_address=self._initiator_address), metadata=({'x-address': self._advertise_address, 'x-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
	
	@override
	async def SuggestJoinZone(self, request: SuggestJoinZoneRequest, context: ServicerContext[SuggestJoinZoneRequest, Empty]) -> Empty:
		request_metadata = dict(context.invocation_metadata())
		request_address = request_metadata['x-address']
		
		async with self._suggestee_peers_lock:
			self._suggestee_peers.add(request_address)
		
		if self._initiator_address is not None:
			if request.initiator_address != self._initiator_address and request.initiator_address not in self._informed_initiators:  # If not got suggested my own initiator, and it is not a previously informed initiator.
				async with self._informed_initiators_lock:
					if request.initiator_address not in self._informed_initiators:
						self._informed_initiators.add(request.initiator_address)
					else:
						return Empty()
				
				async with asyncio.TaskGroup() as tg:
					tg.create_task(self._reject_initiator(request.initiator_address))
					tg.create_task(self._inform_self_initiator(request.initiator_address))
			else:
				pass  # NOP
		else:
			released = False
			try:
				await self._join_zone_lock.acquire()
				
				if self._initiator_address is not None:
					released = True
					self._join_zone_lock.release()
					
					return await self.SuggestJoinZone(request, context)
				else:
					self._initiator_address = request.initiator_address
					
					released = True
					self._join_zone_lock.release()
					
					await self._accept_initiator()
					
					self._no_probe_watchdog_task.cancel('Joined a zone.')
					
					# Suggest the initiator to my peers except the initiator itself and the suggester peer (get the suggester peer from the request's context).
					async with asyncio.TaskGroup() as tg:
						peers = self._peers_addresses - {self._initiator_address, request_metadata['x-address']}
						async with self._suggestee_peers_lock:
							peers -= self._suggestee_peers
						
						for peer_address in peers:
							tg.create_task(self._suggest_initiator(peer_address))
			finally:
				if not released:
					self._join_zone_lock.release()
		
		return Empty()
	
	@override
	async def JoinZone(self, request: JoinZoneRequest, context: ServicerContext[JoinZoneRequest, Empty]) -> Empty:
		assert self._is_initiator_v
		
		request_metadata = dict(context.invocation_metadata())
		request_address = request_metadata['x-address']
		
		match request.decision:
			case JoinZoneDecision.ACCEPT:
				async with self._zone_data_lock:
					assert request_address not in self._zone_data
					
					for peer_address, _ in request.data.peers_addresses_data.items():
						assert peer_address not in self._flattened_zone_data
					
					self._zone_data[request_address] = {k: pickle.loads(v) for k, v in request.data.peers_addresses_data.items()}
					self._invalidate_flattened_zone_data_cache()
				
				self._logger.debug(f"Peer `{request_address}` joined my zone.")
			case JoinZoneDecision.REJECT:
				self._neighbor_initiators_addresses.add(request.initiator_address)
				
				self._logger.debug(f"Peer `{request_address}` rejected joining my zone.")
			case JoinZoneDecision.INFORM:
				self._neighbor_initiators_addresses.add(request.initiator_address)
				
				self._logger.debug(f"Got informed of the neighbor zone initiator `{request.initiator_address}`.")
			case _:
				raise ValueError(f"Invalid `JoinZoneDecision` enum value \"{request.decision}\".")
		
		return Empty()
	
	@override
	async def MigrateZoneInitiator(self, request: MigrateZoneInitiatorRequest, context: ServicerContext[MigrateZoneInitiatorRequest, Empty]) -> Empty:
		request_metadata = dict(context.invocation_metadata())
		request_address = request_metadata['x-address']
		
		assert request_address == self._initiator_address
		
		# noinspection PyAttributeOutsideInit
		self._initiator_address = request.new_zone_initiator_address
		self._logger.debug(f"Migrated to the new zone initiator `{request.new_zone_initiator_address}`.")
		
		return Empty()
	
	@override
	async def PromoteToZoneInitiator(self, request: PromoteToZoneInitiatorRequest, context: ServicerContext[PromoteToZoneInitiatorRequest, Empty]) -> Empty:
		request_metadata = dict(context.invocation_metadata())
		request_address = request_metadata['x-address']
		
		assert self._initiator_address is not None
		assert self._initiator_address == request_address
		assert not self._is_initiator_v
		
		self._logger.info("Selected to be a zone initiator (by a promotion).", extra={'type': 'zone-initiator'})
		
		await self._init_initiator()
		
		self._zone_data.update({peer_address: {peer_address: pickle.loads(datum) for peer_address, datum in data.peers_addresses_data.items()} for peer_address, data in request.peers_addresses_data.items()})
		self._neighbor_initiators_addresses.add(request_address)
		
		# noinspection PyAttributeOutsideInit
		self._is_initiator_v = True
		self._logger.debug(f"Promoted to a zone initiator.")
		
		try:
			from opentelemetry import trace, context
		except ImportError:
			self._tasks.append(asyncio.create_task(self._become_initiator(skip_suggesting=True, skip_partitioning=True)))
		else:
			async def coro():
				with trace.use_span(None, end_on_exit=False):
					await self._become_initiator(skip_suggesting=True, skip_partitioning=True)
			
			self._tasks.append(asyncio.create_task(coro()))
		
		return Empty()
	
	# noinspection PyAttributeOutsideInit
	@override
	async def SuggestClusterRepresentative(self, request: SuggestClusterRepresentativeRequest, context: ServicerContext[SuggestClusterRepresentativeRequest, SuggestClusterRepresentativeResponse]) -> SuggestClusterRepresentativeResponse:
		if request.force:
			self._cluster_datum = pickle.loads(request.data)
			self._cluster_peers_addresses = set(request.peers_addresses)
			self._is_cluster_representative = True
			self._is_super_cluster_representative = False
			
			self._logger.info(f"Accepted the cluster representative role of the {self._cluster_peers_addresses} cluster.", extra={'type': 'cluster', 'peers': list(self._cluster_peers_addresses)})
			
			if not self._is_initiator_v:  # If not one with a higher precedence is willing to build a next level node, which currently that would be an initiator.
				# TODO: this may end-up as a dangling meta node if it does not actually get suggested a cluster representative role in the next level.
				await self._proceed_representative_next_level()
			
			return SuggestClusterRepresentativeResponse(accept=True)
		else:
			raise NotImplementedError
	
	@override
	async def PromoteToSuperClusterRepresentative(self, request, context):
		assert self._is_cluster_representative
		
		# noinspection PyAttributeOutsideInit
		self._super_clusters_representatives_addresses = frozenset(request.super_clusters_representatives_addresses)
		# noinspection PyAttributeOutsideInit
		self._is_super_cluster_representative = True
		
		self._logger.info("Got promoted to a super cluster representative.", extra={'type': 'super-cluster-representative', 'super-clusters-representatives': list(request.super_clusters_representatives_addresses)})
		
		# TODO: can return here and do the rest in the background.
		
		self._logger.debug("Advertising self as the search cluster representative...")
		async with asyncio.TaskGroup() as tg:
			for peer_address in self._cluster_peers_addresses:
				tg.create_task(self._advertise_search_cluster_representative_to_peer(peer_address, NodeIdentity(address=self._advertise_address, level=self._level), level=max(self._level - 1, 0)))
		self._logger.debug("Advertised self as the search cluster representative.")
		
		return Empty()
	
	async def _advertise_search_cluster_representative_to_peer(self, peer_address: str, search_cluster_representative_identity: NodeIdentity, level: int | None = None) -> None:
		if level is None:
			level = self._level
		
		peer_stub = await Node._get_stub(peer_address)
		await peer_stub.AdvertiseSearchClusterRepresentative(search_cluster_representative_identity, metadata=({'x-address': self._advertise_address, 'x-level': str(level)} | ({'x-session': self.session} if self.session is not None else {})).items())
	
	# noinspection PyAttributeOutsideInit
	@override
	async def AdvertiseSearchClusterRepresentative(self, request: NodeIdentity, context: ServicerContext[NodeIdentity, Empty]) -> Empty:
		if not self._is_cluster_representative and self._level != 0:
			return Empty()
		
		if self._level == 0:
			self._search_cluster_representative_address = request.address
			self._search_cluster_representative_level = request.level
			self._search_cluster_representative_event.set()
			
			self._logger.debug(f"Stored the advertised search cluster representative: {self._search_cluster_representative_address}#{self._search_cluster_representative_level}.")
		
		if self._is_cluster_representative:
			self._logger.debug("Propagating the advertised search cluster representative...")
			
			async with asyncio.TaskGroup() as tg:
				for peer_address in self._cluster_peers_addresses - ({self._advertise_address} if self._level == 0 else set()):
					tg.create_task(self._advertise_search_cluster_representative_to_peer(peer_address, request, level=max(self._level - 1, 0)))
			
			self._logger.debug("Propagated the advertised search cluster representative.")
		
		return Empty()
	
	@override
	async def Search(self, request: SearchRequest, context: ServicerContext[SearchRequest, SearchResponse]) -> SearchResponse:
		assert self._is_cluster_representative or self._level == 0
		
		request_metadata = dict(context.invocation_metadata())
		request_datum = pickle.loads(request.datum)
		
		res = {}
		
		if request.sequential_top_k:
			assert self._level > 0 or self._is_cluster_representative  # This variant of the function call would never reach the level #0 peer.
			assert request.k > 0
			
			remaining_k = request.k
			
			if (self._is_cluster_representative and self._is_super_cluster_representative) and ('x-super' in request_metadata and bool(request_metadata['x-super'])):
				# Act as a super cluster representative.
				
				addrs_sims = {}
				async with asyncio.TaskGroup() as tg:
					self_sim_task = tg.create_task(self._similarity(request_datum, self._cluster_datum))
					for addr in self._super_clusters_representatives_addresses:
						addrs_sims[addr] = tg.create_task(self._immediate_search(addr, self._level, request.datum))
					
					for addr, task in addrs_sims.items():
						addrs_sims[addr] = (await task).score
					
					addrs_sims[self._advertise_address] = await self_sim_task
				
				# TODO: optimize the pruning by first sorting.
				addrs_sims = [(addr, sim) for addr, sim in addrs_sims.items() if sim >= request.min_similarity]
				addrs_sims.sort(key=lambda addr_sim: addr_sim[1], reverse=True)
				
				for addr, _ in addrs_sims:
					search_res = await self._search_representative(addr, self._level, SearchRequest(datum=request.datum, k=remaining_k, min_similarity=request.min_similarity, sequential_top_k=True))
					
					res.update(search_res.scores)
					remaining_k -= len(search_res.scores)
					if remaining_k == 0:
						break
			elif self._is_cluster_representative:
				# Act as a cluster representative.
				
				addrs_sims = {}
				async with asyncio.TaskGroup() as tg:
					for addr in self._cluster_peers_addresses:
						addrs_sims[addr] = tg.create_task(self._immediate_search(addr, max(self._level - 1, 0), request.datum))
				
				for addr, task in addrs_sims.items():
					addrs_sims[addr] = (await task).score
				
				# TODO: optimize the pruning by first sorting.
				addrs_sims = [(addr, sim) for addr, sim in addrs_sims.items() if sim >= request.min_similarity]
				addrs_sims.sort(key=lambda addr_sim: addr_sim[1], reverse=True)
				
				if self._level > 0:
					for addr, _ in addrs_sims:
						search_res = await self._search_representative(addr, self._level - 1, SearchRequest(datum=request.datum, k=remaining_k, min_similarity=request.min_similarity, sequential_top_k=True))
						
						res.update(search_res.scores)
						remaining_k -= len(search_res.scores)
						if remaining_k == 0:
							break
				else:
					res.update(addrs_sims[:remaining_k])
			else:
				assert False
		else:
			if self._level == 0:
				similarity = await self._similarity(request_datum, self._data[self._advertise_address])
				if similarity >= request.min_similarity:
					res[self._advertise_address] = similarity
			
			if self._is_cluster_representative:
				async with asyncio.TaskGroup() as tg:
					addrs_tasks = []
					
					request_copy = SearchRequest()
					request_copy.MergeFrom(request)
					
					if self._is_super_cluster_representative and request_metadata['x-address'] not in self._super_clusters_representatives_addresses:
						for addr in self._super_clusters_representatives_addresses:
							addrs_tasks.append(tg.create_task(self._search_representative(addr, self._level, request_copy)))
					
					cluster_similarity = await self._similarity(request_datum, self._cluster_datum)
					if cluster_similarity >= request.min_similarity:
						for cluster_peer_address in self._cluster_peers_addresses - ({self._advertise_address} if self._level == 0 else set()):
							addrs_tasks.append(tg.create_task(self._search_representative(cluster_peer_address, max(self._level - 1, 0), request_copy)))
					
					# Release the search data while waiting for I/O.
					request.ClearField('datum')
					del request_datum
					del request_copy
					
					# TODO: migrate to `async for` after updating to Python 3.13.
					for t in asyncio.as_completed(addrs_tasks):
						t_res = await t
						res.update(t_res.scores)
			
			if request.k > 0:
				res = dict(list(sorted(res.items(), key=lambda peer_address_score: peer_address_score[1], reverse=True))[:request.k])
		
		return SearchResponse(scores=res)
	
	# noinspection PyTypeChecker
	@override
	async def ImmediateSearch(self, request: ImmediateSearchRequest, context: ServicerContext[ImmediateSearchRequest, ImmediateSearchResponse]) -> ImmediateSearchResponse:
		request_metadata = dict(context.invocation_metadata())
		request_from_level = int(request_metadata.get('x-from-level', 0))
		
		assert self._is_cluster_representative or (self._level == 0 and request_from_level == 0)
		
		request_datum = pickle.loads(request.datum)
		
		if self._is_cluster_representative:
			similarity = await self._similarity(request_datum, self._cluster_datum)
		elif self._level == 0 and request_from_level == 0:
			similarity = await self._similarity(request_datum, self._data[self._advertise_address])
		else:
			assert False
		
		return ImmediateSearchResponse(score=similarity)
	
	# noinspection PyMethodMayBeStatic
	async def _search_representative(self, address: str, level: int, request: SearchRequest, super_: bool = False) -> SearchResponse:
		# TODO: optimize when `addr == self._advertise_address` by not going into the network stack.
		stub = await Node._get_stub(address)
		return await stub.Search(request, metadata=({'x-level': str(level), 'x-address': self._advertise_address} | ({'x-session': self.session} if self.session is not None else {}) | ({'x-super': str(super_)} if super_ else {})).items())
	
	async def _immediate_search(self, address: str, level: int, datum: bytes) -> ImmediateSearchResponse:
		# TODO: optimize when `addr == self._advertise_address` by not going into the network stack.
		stub = await Node._get_stub(address)
		return await stub.ImmediateSearch(ImmediateSearchRequest(datum=datum), metadata=({'x-level': str(level), 'x-address': self._advertise_address, 'x-from-level': str(self._level)} | ({'x-session': self.session} if self.session is not None else {})).items())
	
	async def wait_ready(self) -> None:
		await self._search_cluster_representative_event.wait()
	
	async def search(self, datum: Any, min_similarity: float, k: int = 0, sequential_top_k: bool = False) -> dict[str, float]:
		res = await self._search_representative(
			address=self._search_cluster_representative_address, level=self._search_cluster_representative_level, super_=True,
			request=SearchRequest(datum=pickle.dumps(datum), k=k, min_similarity=min_similarity, sequential_top_k=sequential_top_k),
		)
		return dict(res.scores)
	
	def __await__(self):
		async def _coro():
			await self.__ainit__()
			return self
		
		return _coro().__await__()
	
	grpc_options = {  # Default gRPC options for the server and the channels.
		'grpc.max_receive_message_length': -1,
		'grpc.max_send_message_length': -1
	}
	
	async def close(self, release_stubs: bool = True) -> None:
		pass  # FIXME: close next levels' nodes first.
		
		async with asyncio.TaskGroup() as tg:
			# noinspection PySimplifyBooleanCheck
			if self._init_grpc_server == True:
				tg.create_task(self._grpc_server.stop(grace=sys.float_info.max))
			if release_stubs:
				tg.create_task(Node.release_all_stubs())
			if not self._is_initiator_v:
				self._no_probe_watchdog_task.cancel('Closing...')
			
			async def _cancel_and_wait_task(t: asyncio.Task) -> None:
				t.cancel('Closing...')
				
				try:
					await t
				except BaseException as e:
					traceback.print_exception(e)
			
			while len(self._tasks) > 0:
				tg.create_task(_cancel_and_wait_task(self._tasks.pop(0)))
	
	_stubs: dict[str, tuple[Stub, Channel]] = dict()
	_stubs_lock = asyncio.Lock()
	
	@staticmethod
	async def _get_stub(address: str) -> Stub:
		if ':' not in address:
			address = f'{address}:50051'
		
		stub, _ = Node._stubs.get(address, (None, None))
		if stub is not None:
			return stub
		
		async with Node._stubs_lock:
			stub, _ = Node._stubs.get(address, (None, None))
			if stub is not None:
				return stub
			
			channel = insecure_channel(target=address, options=list(Node.grpc_options.items()))
			
			stub = Stub(channel)
			Node._stubs[address] = (stub, channel)
			return stub
	
	@staticmethod
	async def _release_stub(address: str) -> None:
		if ':' not in address:
			address = f'{address}:50051'
		
		# TODO: undefined behavior if the stub/channel is being used (a reference to it is held).
		
		_, channel = Node._stubs.get(address, (None, None))
		await channel.close()
		del Node._stubs[address]
	
	@staticmethod
	async def release_all_stubs():
		async with Node._stubs_lock:
			async with asyncio.TaskGroup() as tg:
				for address in Node._stubs.keys():
					tg.create_task(Node._release_stub(address))


class MetadataMultiplexingServicer(Servicer):
	def __init__(self, metadata_key: str, metadata_value_transform: Callable[[str], Hashable] = lambda v: v, allow_replacement: bool = True) -> None:
		self._metadata_key = metadata_key
		self._metadata_value_transform = metadata_value_transform
		self._allow_replacement = allow_replacement
		
		self._servicers = dict[Hashable, Servicer | asyncio.Event]()
		self._servicers_lock = asyncio.Lock()
	
	async def register_servicer(self, metadata_value: Hashable, servicer: Servicer) -> None:
		assert isinstance(servicer, Servicer)
		
		async with self._servicers_lock:
			event = self._servicers.get(metadata_value, None)
			
			self._servicers[metadata_value] = servicer
			
			if event is not None:
				if isinstance(event, asyncio.Event):
					event.set()
				else:
					if self._allow_replacement:
						pass  # NOP.
					else:
						raise ValueError(f'Servicer {self._metadata_key}={metadata_value} already exists and replacement is not allowed.')
	
	async def unregister_servicer(self, metadata_value: Hashable) -> None:
		async with self._servicers_lock:
			servicer = self._servicers.get(metadata_value, None)
			if isinstance(servicer, Servicer):
				del self._servicers[metadata_value]
			else:
				# Probably `asyncio.Event` or `None`.
				raise ValueError(f"No servicer for the metadata {self._metadata_key}={metadata_value} is registered.")
	
	async def _servicer(self, metadata_value: Hashable) -> Servicer:
		servicer = self._servicers.get(metadata_value, None)
		if servicer is None:
			async with self._servicers_lock:
				servicer = self._servicers.get(metadata_value, None)
				if servicer is None:
					servicer = asyncio.Event()
					self._servicers[metadata_value] = servicer
		
		if isinstance(servicer, asyncio.Event):
			_logger.debug(f"Waiting for the {self._metadata_key}={metadata_value} servicer...")
			await servicer.wait()
			return await self._servicer(metadata_value)
		
		return servicer
	
	async def _context_servicer(self, context: ServicerContext[Any, Any]) -> Servicer:
		metadata_value = self._metadata_value_transform(dict(context.invocation_metadata())[self._metadata_key])
		return await self._servicer(metadata_value)
	
	@override
	async def SuggestJoinZone(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.SuggestJoinZone(request, context)
	
	@override
	async def SuggestClusterRepresentative(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.SuggestClusterRepresentative(request, context)
	
	@override
	async def MigrateZoneInitiator(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.MigrateZoneInitiator(request, context)
	
	@override
	async def PromoteToZoneInitiator(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.PromoteToZoneInitiator(request, context)
	
	@override
	async def JoinZone(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.JoinZone(request, context)
	
	@override
	async def Search(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.Search(request, context)
	
	@override
	async def PromoteToSuperClusterRepresentative(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.PromoteToSuperClusterRepresentative(request, context)
	
	@override
	async def AdvertiseSearchClusterRepresentative(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.AdvertiseSearchClusterRepresentative(request, context)
	
	@override
	async def ImmediateSearch(self, request, context: ServicerContext):
		servicer = await self._context_servicer(context)
		return await servicer.ImmediateSearch(request, context)


class SessionMultiplexingServicer(MetadataMultiplexingServicer):
	def __init__(self):
		super().__init__(metadata_key='x-session')
	
	async def register_session_servicer(self, session: str, servicer: Servicer) -> None:
		await self.register_servicer(session, servicer)
	
	async def unregister_session_servicer(self, session: str) -> None:
		await self.unregister_servicer(session)


class LevelMultiplexingServicer(MetadataMultiplexingServicer):
	def __init__(self) -> None:
		super().__init__(metadata_key='x-level', metadata_value_transform=lambda v: int(v), allow_replacement=False)
	
	async def register_level_servicer(self, level: int, servicer: Servicer) -> None:
		await self.register_servicer(level, servicer)
	
	@override
	async def unregister_servicer(self, metadata_value: Hashable) -> None:
		raise NotImplementedError
