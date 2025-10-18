import asyncio
import ipaddress
import logging
import random
import sys
from collections import OrderedDict
from datetime import timedelta

from p2psims import Node
from tabulate import tabulate

logging.basicConfig(
	format="{name}\t{asctime}\t{levelname}\t{message}\t", style='{',
	level=logging.DEBUG
)

NODES_COUNT = 20
NODES_PER_ZONE_INITIATOR = 5


async def node_amain(i: int):
	port = 50051 + i
	address = f"localhost:{port}"
	peers_addresses = {f"localhost:{50051 + i}" for i in range(NODES_COUNT)} - {address}
	
	node = Node(
		data=[random.randint(0, 100) for i in range(5)],
		initial_peers_addresses=random.sample(list(peers_addresses), NODES_COUNT // NODES_PER_ZONE_INITIATOR),
		preferred_zone_size=NODES_PER_ZONE_INITIATOR,
		grpc_server_listen_address=f"[::]:{port}",
		advertise_address=address,
		advertise_ip=ipaddress.IPv4Address(f"127.0.0.{i}"),
		no_probe_duration=timedelta(seconds=1),
	)
	try:
		await node
		
		if i == 0:
			await node.wait_ready()
			
			search_res = await node.search([0, 1, 2, 3, 4], k=7, min_similarity=0, sequential_top_k=True)
			search_res = OrderedDict(sorted(search_res.items(), key=lambda it: it[1], reverse=True))
			
			print(tabulate({'Peer': search_res.keys(), 'Similarity': search_res.values()}, headers='keys', tablefmt='grid'))
		
		await asyncio.sleep(sys.float_info.max)
	finally:
		await node.close()


async def amain():
	async with asyncio.TaskGroup() as tg:
		for i in range(NODES_COUNT):
			tg.create_task(node_amain(i))


asyncio.run(amain())
