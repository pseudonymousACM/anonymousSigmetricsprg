import functools
import random
from contextlib import contextmanager

import kmedoids
import numpy as np
import sklearn.cluster
import torch

import conf

torch.utils.deterministic.fill_uninitialized_memory = True  # This is the default, but it is set explicitly for maximum clarity.

if conf.TORCH_DET_ALG:
	torch.use_deterministic_algorithms(True)

if conf.TORCH_DET_CUDA and torch.cuda.is_available():
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def _seed(x):
	random.seed(x)
	np.random.seed(x)
	torch.manual_seed(x)


_seed(conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)

# noinspection PyUnresolvedReferences
sklearn.cluster.AffinityPropagation = functools.partial(sklearn.cluster.AffinityPropagation, random_state=conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)
# noinspection PyUnresolvedReferences
sklearn.cluster.KMeans = functools.partial(sklearn.cluster.KMeans, random_state=conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)
kmedoids.KMedoids = functools.partial(kmedoids.KMedoids, random_state=conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)


# This only works if no other threads have a conflicting interest in the global RNGs; TODO: find a thread local work-around for it.
@contextmanager
def seed(x):
	if torch.cuda.is_available():
		torch_cuda_rng_states = torch.cuda.get_rng_state_all()
	
	torch_cpu_rng_state = torch.get_rng_state()
	np_random_rng_state = np.random.get_state()
	random_rng_state = random.getstate()
	
	_seed(x)
	
	try:
		yield
	finally:
		random.setstate(random_rng_state)
		np.random.set_state(np_random_rng_state)
		torch.set_rng_state(torch_cpu_rng_state)
		
		if torch.cuda.is_available():
			# noinspection PyUnboundLocalVariable
			torch.cuda.set_rng_state_all(torch_cuda_rng_states)
