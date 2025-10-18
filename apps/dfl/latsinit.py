import asyncio
import logging
import sys
from random import Random
from typing import Iterable

import networkx as nx

_INF_RATE = '99999999tbit'
_FAIR_QUANTUM = 1500

_tc_id = 1


# TODO: move this function into an extensions/utils package.
async def _sh(s: str, return_stdout: bool = False) -> str | None:
	proc = await asyncio.create_subprocess_shell(s, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
	stdout, stderr = await proc.communicate()
	
	stderr = stderr.decode().strip()
	stdout = stdout.decode().strip()
	
	if stderr != '':
		print(stderr, file=sys.stderr, flush=True)
	if stdout != '' and not return_stdout:
		print(stdout, flush=True)
	
	assert proc.returncode == 0
	
	if return_stdout:
		return stdout
	else:
		return None


async def _init():
	global _tc_id
	
	await _sh(f"tc qdisc add dev eth0 root handle 1: htb default {_tc_id} && "
	          f"tc class add dev eth0 parent 1: classid 1:{_tc_id} htb rate {_INF_RATE} quantum {_FAIR_QUANTUM}")
	
	_tc_id += 1


async def _apply(peer: str, latency_ms: int, tc_id: int | None = None) -> None:
	global _tc_id
	
	if tc_id is None:
		tc_id = _tc_id
		_tc_id += 1
	
	ip = None
	async with asyncio.TaskGroup() as tg:
		async def _coro1():
			nonlocal ip
			
			ip = await _sh(f"dig +short {peer}", return_stdout=True)
			await _sh(f"iptables -t mangle -A OUTPUT -d \"{ip}\" -j MARK --set-mark {tc_id}")
		
		async def _coro2():
			nonlocal ip
			
			await _sh(f"tc class add dev eth0 parent 1: classid 1:{tc_id} htb rate {_INF_RATE} quantum {_FAIR_QUANTUM}")
			await _sh(f"tc qdisc add dev eth0 parent 1:{tc_id} handle {tc_id}: netem delay {latency_ms}ms")
			await _sh(f"tc filter add dev eth0 protocol ip parent 1:0 prio 1 handle {tc_id} fw flowid 1:{tc_id}")
		
		tg.create_task(_coro1())
		tg.create_task(_coro2())
	
	logging.info(f"Simulating {latency_ms} ms latency (one-way) to {peer}/{ip} with tc-id {tc_id}...", extra={'type': 'simulate-latency', 'peer': peer, 'latency-ms': latency_ms, 'tc_id': tc_id, 'ip': ip})


async def initial_peers(g: nx.Graph, self_index: int, rnd: Random | None = None, rnd_range: tuple[int, int] = (0, 100)) -> None:
	global _tc_id
	
	await _init()
	
	if rnd is None:
		rnd = Random(42)
	
	for u, v in g.edges():
		if 'latency_ms' not in g[u][v]:
			# noinspection PyTypeChecker
			g[u][v]['latency_ms'] = rnd.randint(rnd_range[0], rnd_range[1])
	
	shortest_paths = nx.single_source_dijkstra_path_length(g, self_index, weight='latency_ms')
	
	async with asyncio.TaskGroup() as tg:
		for peer_index, latency_ms in shortest_paths.items():
			if peer_index == self_index:
				continue
			
			peer_name = f'n{peer_index}'
			tg.create_task(_apply(peer_name, latency_ms // 2, tc_id=_tc_id))
			_tc_id += 1


async def initial_peers_local(g: nx.Graph, self_index: int, rnd: Random | None = None, rnd_range: tuple[int, int] = (0, 100)) -> None:
	global _tc_id
	
	await _init()
	
	if rnd is None:
		rnd = Random(42)
	
	for u, v in g.edges():
		if 'latency_ms' not in g[u][v]:
			# noinspection PyTypeChecker
			g[u][v]['latency_ms'] = rnd.randint(rnd_range[0], rnd_range[1])
	
	async with asyncio.TaskGroup() as tg:
		for peer_index in g.neighbors(self_index):
			if peer_index == self_index:
				continue
			
			latency_ms = g[self_index][peer_index]['latency_ms']
			
			peer_name = f'n{peer_index}'
			# noinspection PyTypeChecker
			tg.create_task(_apply(peer_name, latency_ms // 2, tc_id=_tc_id))
			_tc_id += 1


async def list_(peers_latencies: dict[str, int]) -> None:
	global _tc_id
	
	await _init()
	
	async with asyncio.TaskGroup() as tg:
		for peer, latency in peers_latencies.items():
			tg.create_task(_apply(peer, latency, tc_id=_tc_id))
			_tc_id += 1


async def all_random(all_peers: Iterable[str], rnd: Random | None = None, rnd_range: tuple[int, int] = (0, 100)) -> None:
	global _tc_id
	
	if rnd is None:
		rnd = Random(42)
	
	await _init()
	
	async with asyncio.TaskGroup() as tg:
		for peer in all_peers:
			latency = rnd.randint(rnd_range[0] // 2, rnd_range[1] // 2)
			tg.create_task(_apply(peer, latency, tc_id=_tc_id))
			_tc_id += 1
