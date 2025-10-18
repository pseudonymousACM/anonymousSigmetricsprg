import platform
import re
from os import environ

NAME = environ.get('NAME', platform.node())

GLOBAL_RANDOM_SEED = int(environ.get('GLOBAL_RANDOM_SEED', '42'))
assert GLOBAL_RANDOM_SEED != 0

NODE_GLOBAL_RANDOM_SEED = environ.get('NODE_GLOBAL_RANDOM_SEED', None)
if NODE_GLOBAL_RANDOM_SEED is None:
	NODE_GLOBAL_RANDOM_SEED = GLOBAL_RANDOM_SEED
else:
	NODE_GLOBAL_RANDOM_SEED = int(NODE_GLOBAL_RANDOM_SEED)
assert NODE_GLOBAL_RANDOM_SEED != 0


# TODO: move this into a extensions/utils package.
# TODO: also add and use a node index to name function.
def _node_name_to_index(name: str) -> int | None:
	# Use the first number inside the name as the index.
	match_ = re.search(r'\d+', name)
	index = int(match_.group()) if match_ else None
	return index


INDEX = environ.get('INDEX', None)
if INDEX is None:
	INDEX = _node_name_to_index(NAME)
	if INDEX is None:
		raise ValueError(f"Cannot determine the node index from its name \"{NAME}\".")
else:
	INDEX = int(INDEX)

NODES_COUNT = int(environ.get('NODES_COUNT', '10'))  # Copy-pasta-used in `./compose-gen/main.py`; the default value should match with there.

DATA_DISTRIBUTION_NODES_COUNT = environ.get('DATA_DISTRIBUTION_NODES_COUNT', None)
if DATA_DISTRIBUTION_NODES_COUNT is not None:
	DATA_DISTRIBUTION_NODES_COUNT = int(DATA_DISTRIBUTION_NODES_COUNT)
else:
	DATA_DISTRIBUTION_NODES_COUNT = NODES_COUNT

ALL_NODES_LIST = [f'n{i}' for i in range(NODES_COUNT)]

ALL_NODES = set(ALL_NODES_LIST)

ALL_PEERS = ALL_NODES - {NAME}

if INDEX < 0 or INDEX > NODES_COUNT:
	raise ValueError(f"Node index {INDEX} is out of range [0, NODES_COUNT={NODES_COUNT}).")

DEVICE = environ.get('DEVICE', None)  # Copy-pasta-used in `./compose-gen/main.py`.

DATA_DISTRIBUTION = environ.get('DATA_DISTRIBUTION', 'stratified').lower()
assert DATA_DISTRIBUTION in ('stratified', 'one-ten', 'iid', 'non-iid-strict-label-dir', 'non-iid-label-dir', 'iid-strict-dir-quantity', 'iid-dir-quantity', 'prepared-file', 'all') \
       or DATA_DISTRIBUTION.startswith('mojtaba-non-iid-') \
       or DATA_DISTRIBUTION.startswith('non-iid-strict-label-dir-') \
       or DATA_DISTRIBUTION.startswith('non-iid-label-dir-') \
       or DATA_DISTRIBUTION.startswith('non-iid-') \
       or DATA_DISTRIBUTION.startswith('fixed-') \
       or DATA_DISTRIBUTION.startswith('pretty-non-iid-') \
       or DATA_DISTRIBUTION.startswith('iid-strict-dir-quantity-') \
       or DATA_DISTRIBUTION.startswith('iid-dir-quantity-') \
       or DATA_DISTRIBUTION.startswith('pathological-non-iid-')

DATA = environ.get('DATA', 'mnist').lower()
assert DATA in ('mnist', 'cifar10', 'emnist', 'fmnist', 'cifar100', 'femnist') \
       or DATA.startswith('emnist-')

MODEL = environ.get('MODEL', '1nn').lower()
assert MODEL in ('1nn', '2nn', 'cnn', 'resnet18')

ROUNDS = int(environ.get('ROUNDS', '10'))

CLUSTERING = environ.get('CLUSTERING', 'affinity-propagation').lower()
assert CLUSTERING in ('affinity-propagation', 'affinity-propagation-euclidean', 'affinity-propagation-cosine') \
       or CLUSTERING.startswith('kmeans-') \
       or CLUSTERING.startswith('kmedoids-')

SIMILARITY = environ.get('SIMILARITY', 'cosine').lower()
assert SIMILARITY in ('cosine', 'euclidean', 'cosine-raw')


# TODO: move this into an extension package.
def _is_float(s) -> bool:
	try:
		float(s)
		return True
	except ValueError:
		return False


PEERS_SELECTION = environ.get('PEERS_SELECTION', f'ideal-top-{max(NODES_COUNT // 3, 1)}').lower()
assert PEERS_SELECTION in ('everyone', 'no-one', 'initial-peers') \
       or PEERS_SELECTION.startswith('ideal-top-') \
       or PEERS_SELECTION.startswith('min-similarity-') \
       or PEERS_SELECTION.startswith('uniform-random-') \
       or PEERS_SELECTION.startswith('top-') \
       or PEERS_SELECTION.startswith('parallel-topk-isotonic-') \
       or PEERS_SELECTION.startswith('par-topk-lin-reg-') \
       or PEERS_SELECTION.startswith('par-topk-ideal-th-')

SEARCH_FUL_PEERS_SELECTION = any(PEERS_SELECTION.startswith(s) for s in ('min-similarity-', 'top-', 'parallel-topk-isotonic-', 'par-topk-lin-reg-', 'par-topk-ideal-th-'))

PULL_EARLY_PEERS_SELECTION = PEERS_SELECTION in ('everyone', 'no-one', 'initial-peers') \
                             or PEERS_SELECTION.startswith('uniform-random-')

SIMILARITY_BASED_PEERS_SELECTION = PEERS_SELECTION.startswith('ideal-top-') \
                                   or PEERS_SELECTION.startswith('min-similarity-') \
                                   or PEERS_SELECTION.startswith('top-') \
                                   or PEERS_SELECTION.startswith('parallel-topk-isotonic-') \
                                   or PEERS_SELECTION.startswith('par-topk-lin-reg-') \
                                   or PEERS_SELECTION.startswith('par-topk-ideal-th-')

EPOCHS = int(environ.get('EPOCHS', '1'))
assert EPOCHS >= 0

BATCH_SIZE = int(environ.get('BATCH_SIZE', '100'))
assert BATCH_SIZE >= 1

LR = environ.get('LR', None)
if LR is not None:
	LR = float(LR)

SEARCH_DATA = environ.get('SEARCH_DATA', 'params').lower()
assert SEARCH_DATA in ('params', 'grads') \
       or SEARCH_DATA.startswith('params-top-') \
       or SEARCH_DATA.startswith('grads-top-') \
       or SEARCH_DATA.startswith('params-grads-top-')

GRADS_BASED_SEARCH_DATA = SEARCH_DATA == 'grads' \
                          or SEARCH_DATA.startswith('grads-top-') \
                          or SEARCH_DATA.startswith('params-grads-top-')

INITIAL_PEERS = environ.get('INITIAL_PEERS', 'everyone').lower()
assert INITIAL_PEERS in ('everyone', 'rgg', 'erg', 'rst', 'bag', 'wsg', 'no-one') \
       or INITIAL_PEERS.startswith('list-') \
       or INITIAL_PEERS.startswith('bag-') \
       or INITIAL_PEERS.startswith('wsg-')

PREFERRED_ZONE_SIZE = int(environ.get('PREFERRED_ZONE_SIZE', str(max(NODES_COUNT // 3, 2))))
assert PREFERRED_ZONE_SIZE >= 2

MAX_ZONE_SIZE = environ.get('MAX_ZONE_SIZE', None)
if MAX_ZONE_SIZE is not None:
	MAX_ZONE_SIZE = int(MAX_ZONE_SIZE)
	if MAX_ZONE_SIZE <= 0:
		MAX_ZONE_SIZE = None
else:
	MAX_ZONE_SIZE = PREFERRED_ZONE_SIZE

SIMULATE_LATENCIES_RANDOM_RANGE = environ.get('SIMULATE_LATENCIES_RANDOM_RANGE', '10,100')
SIMULATE_LATENCIES_RANDOM_RANGE = tuple[int, int](map(int, SIMULATE_LATENCIES_RANDOM_RANGE.split(',', maxsplit=1)))

SIMULATE_LATENCIES = environ.get('SIMULATE_LATENCIES', 'none')
assert SIMULATE_LATENCIES in ('none', 'initial-peers', 'initial-peers-local', 'all-random') \
       or SIMULATE_LATENCIES.startswith('list-')

SEARCH_RENEW_ROUNDS = int(environ.get('SEARCH_RENEW_ROUNDS', '1'))
assert SEARCH_RENEW_ROUNDS >= 1

SYNC_ROUNDS = environ.get('SYNC_ROUNDS', 'True').lower() in ('true', 'y', 'yes', '1')

ROUNDS_AGE = environ.get('ROUNDS_AGE', None)
if ROUNDS_AGE is None:
	if SYNC_ROUNDS:
		ROUNDS_AGE = 0
	elif SEARCH_FUL_PEERS_SELECTION:
		ROUNDS_AGE = SEARCH_RENEW_ROUNDS
	else:
		ROUNDS_AGE = 1
else:
	ROUNDS_AGE = int(ROUNDS_AGE)

assert ROUNDS_AGE >= 0

GC_PER_ROUND = environ.get('GC_PER_ROUND', 'True').lower() in ('true', 'y', 'yes', '1')

PRENUP_ROUND = environ.get('PRENUP_ROUND', 'True').lower() in ('true', 'y', 'yes', '1')

SEARCH_NO_PROBE_DURATION = float(environ.get('SEARCH_NO_PROBE_DURATION', min(NODES_COUNT / 3, 30)))

CENTRALIZED = environ.get('CENTRALIZED', 'false').lower() in ('true', 'y', 'yes', '1')

MONITOR_SIMILARITIES = environ.get('MONITOR_SIMILARITIES', None)
assert MONITOR_SIMILARITIES in (None, 'n0', 'split', 'none')

if MONITOR_SIMILARITIES is None:
	if CENTRALIZED:
		MONITOR_SIMILARITIES = 'none'
	else:
		if SIMILARITY_BASED_PEERS_SELECTION:
			if SYNC_ROUNDS:
				MONITOR_SIMILARITIES = 'n0'
			else:
				MONITOR_SIMILARITIES = 'split'
		else:
			MONITOR_SIMILARITIES = 'none'

TRACE_RPCS = environ.get('TRACE_RPCS', 'False').lower() in ('true', 'y', 'yes', '1')

DATA_PIN_MEMORY = environ.get('DATA_PIN_MEMORY', 'True').lower() in ('true', 'y', 'yes', '1')

OPTIMIZER = environ.get('OPTIMIZER', 'adam').lower()
assert OPTIMIZER in ('adam', 'sgd')

OPTIMIZER_RESET = environ.get('OPTIMIZER_RESET', 'false').lower() in ('true', 'y', 'yes', '1')

HOMO_MODEL_INIT = environ.get('HOMO_MODEL_INIT', 'false').lower() in ('true', 'y', 'yes', '1')

TORCH_INTRA_OP_THREADS = environ.get('TORCH_INTRA_OP_THREADS', None)
if TORCH_INTRA_OP_THREADS not in (None, ''):
	TORCH_INTRA_OP_THREADS = int(TORCH_INTRA_OP_THREADS)
else:
	TORCH_INTRA_OP_THREADS = None

TORCH_INTER_OP_THREADS = environ.get('TORCH_INTER_OP_THREADS', None)
if TORCH_INTER_OP_THREADS not in (None, ''):
	TORCH_INTER_OP_THREADS = int(TORCH_INTER_OP_THREADS)
else:
	TORCH_INTER_OP_THREADS = None

ASYNCIO_IO_THREADS = environ.get('ASYNCIO_IO_THREADS', None)
if ASYNCIO_IO_THREADS not in (None, ''):
	ASYNCIO_IO_THREADS = int(ASYNCIO_IO_THREADS)
else:
	ASYNCIO_IO_THREADS = None

PREPARE_DATA_DISTRIBUTION = len(environ.get('PREPARE_DATA_DISTRIBUTION', '')) > 0

CACHE_SEARCH_DATA = ((PEERS_SELECTION.startswith('ideal-top-') or PEERS_SELECTION.startswith('par-topk-ideal-th-')) and MONITOR_SIMILARITIES) \
                    or MONITOR_SIMILARITIES  # When monitoring both similarities and SON similarities.

CENTRALIZED_AGGREGATOR = environ.get('CENTRALIZED_AGGREGATOR', 'FedAvg').lower()
assert CENTRALIZED_AGGREGATOR in ('fedavg', 'clustering')

LR_DECAY = environ.get('LR_DECAY', None)
if LR_DECAY is not None:
	LR_DECAY = float(LR_DECAY)

AGGREGATOR_DEVICE = environ.get('AGGREGATOR_DEVICE', None)

SYNC_INIT_GRPC_SERVER = environ.get('SYNC_INIT_GRPC_SERVER', 'True').lower() in ('true', 'y', 'yes', '1')
SYNC_EXIT = environ.get('SYNC_EXIT', 'True').lower() in ('true', 'y', 'yes', '1')

TRAIN_BATCH_DROP_LAST = environ.get('TRAIN_BATCH_DROP_LAST', 'False').lower() in ('true', 'y', 'yes', '1')

SYNC_SON_INIT = environ.get('SYNC_SON_INIT', 'True').lower() in ('true', 'y', 'yes', '1')

FIRST_ROUND_TRAIN_EPOCHS = environ.get('FIRST_ROUND_TRAIN_EPOCHS', None)
if FIRST_ROUND_TRAIN_EPOCHS is not None:
	FIRST_ROUND_TRAIN_EPOCHS = int(FIRST_ROUND_TRAIN_EPOCHS)
	assert FIRST_ROUND_TRAIN_EPOCHS >= 0
else:
	FIRST_ROUND_TRAIN_EPOCHS = EPOCHS

CACHE_SON_SEARCH_DATA = PEERS_SELECTION.startswith('par-topk-ideal-th-')

TORCH_DET_CUDA = environ.get('TORCH_DET_CUDA', 'True').lower() in ('true', 'y', 'yes', '1')
TORCH_DET_ALG = environ.get('TORCH_DET_ALG', 'False').lower() in ('true', 'y', 'yes', '1')

SON_DATA_DEVICE = environ.get('SON_DATA_DEVICE', None)  # Or to be known as `SEARCH_DATA_DEVICE`.

STDERR_DEBUG_LOG = environ.get('STDERR_DEBUG_LOG', 'true').lower() in ('true', 'y', 'yes', '1')

DISABLE_CDFL = environ.get('DISABLE_CDFL', 'false').lower() in ('true', 'y', 'yes', '1')

SON_AGG = environ.get('SON_AGG', None)
if SON_AGG is not None:
	SON_AGG = SON_AGG.lower()
	assert SON_AGG in ('mean', 'mean-w', 'mean-w-pnorm')
else:
	match SIMILARITY:
		case 'cosine' | 'cosine-raw':
			SON_AGG = 'mean-w-pnorm'
		case 'euclidean':
			SON_AGG = 'mean-w'
		case _:
			assert False

L2 = environ.get('L2', None)
if L2 is not None:
	L2 = float(L2)
