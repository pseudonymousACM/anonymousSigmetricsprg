import logging
from os import environ
from pathlib import Path
from random import Random

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

import conf
import conf_eval
import data_dist

_logger = logging.getLogger(__name__)

_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

_download = environ.get('DATA_DOWNLOAD', 'False').lower() in {'true', '1', 't', 'y', 'yes'}
match conf.DATA:
	case 'cifar10':
		train_data_all = datasets.CIFAR10(root='./data/', train=True, download=_download, transform=_transform)
		test_data_all = datasets.CIFAR10(root='./data/', train=False, download=_download, transform=_transform)
	case 'mnist':
		train_data_all = datasets.MNIST(root='./data/', train=True, download=_download, transform=_transform)
		test_data_all = datasets.MNIST(root='./data/', train=False, download=_download, transform=_transform)
	case 'fmnist':
		train_data_all = datasets.FashionMNIST(root='./data/', train=True, download=_download, transform=_transform)
		test_data_all = datasets.FashionMNIST(root='./data/', train=False, download=_download, transform=_transform)
	case 'cifar100':
		train_data_all = datasets.CIFAR100(root='./data/', train=True, download=_download, transform=_transform)
		test_data_all = datasets.CIFAR100(root='./data/', train=False, download=_download, transform=_transform)
	case 'femnist':
		if not _download and not any(k in environ for k in ('HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')):
			environ['HF_DATASETS_OFFLINE'] = '1'
		
		from flwr_datasets import FederatedDataset
		from flwr_datasets.partitioner import NaturalIdPartitioner
		
		# TODO: this tries to acquire a lock file for downloading/checking the dataset and the lock file is shared between the containers (resides beside the dataset in the cache directory); also, sends a couple of unnecessary requests checking on the dataset.
		_fds = FederatedDataset(
			dataset='flwrlabs/femnist',
			partitioners={'train': NaturalIdPartitioner(partition_by='writer_id')},
			cache_dir='./data/'
		)
		
		_rng = Random(conf.GLOBAL_RANDOM_SEED)
		_ids = _rng.sample(range(3597), conf.NODES_COUNT)  # Has 3597 partitions.
		
		_my_id = _ids[conf.INDEX]
		_logger.info(f"Chose FEMNIST partition ID {_my_id}.", extra={'type': 'femnist', 'id': _my_id})
		_partition = _fds.load_partition(partition_id=_my_id)
		
		
		def _apply_transforms(batch: dict):
			batch.pop('writer_id', None)
			batch.pop('hsf_id', None)
			
			if 'image' in batch:
				batch['image'] = [_transform(img) for img in batch['image']]
			if 'character' in batch:
				batch['character'] = [torch.tensor(ch) for ch in batch['character']]
			
			return batch
		
		
		_partition.set_transform(_apply_transforms)
		
		_partition_train_test = _partition.train_test_split(test_size=0.2, seed=conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)
		train_data_all, test_data_all = _partition_train_test['train'], _partition_train_test['test']
		
		train_data_all.classes = train_data_all.features['character'].names
		test_data_all.classes = test_data_all.features['character'].names
		
		train_data_all.targets = train_data_all['character']
		test_data_all.targets = test_data_all['character']
		
		
		def _collate_fn(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
			images = torch.stack([item['image'] for item in batch])
			characters = torch.stack([item['character'] for item in batch])
			
			return images, characters
		
		
		del _fds, _partition, _apply_transforms, _partition_train_test, _rng, _ids, _my_id
	case _:
		if conf.DATA == 'emnist' or conf.DATA.startswith('emnist-'):
			_split = 'byclass' if conf.DATA == 'emnist' else conf.DATA.split('-', maxsplit=1)[1]
			train_data_all = datasets.EMNIST(root='./data/', train=True, download=_download, transform=_transform, split=_split)
			test_data_all = datasets.EMNIST(root='./data/', train=False, download=_download, transform=_transform, split=_split)
		else:
			raise ValueError(f"Unknown dataset: {conf.DATA}.")

num_labels = max(len(train_data_all.classes), len(test_data_all.classes))

if conf.NODES_COUNT != conf.DATA_DISTRIBUTION_NODES_COUNT and \
		(
				conf.DATA_DISTRIBUTION in ('one-ten',)
				or conf.DATA_DISTRIBUTION.startswith('pretty-non-iid-')
				or conf.DATA_DISTRIBUTION.startswith('non-iid-')
		):
	assert conf.NODES_COUNT < conf.DATA_DISTRIBUTION_NODES_COUNT
	
	data_dist_nodes_indices = np.round(np.linspace(0, conf.DATA_DISTRIBUTION_NODES_COUNT - 1, conf.NODES_COUNT)).astype(int)
	_data_dist_node_idx = data_dist_nodes_indices[conf.INDEX]
else:
	data_dist_nodes_indices = list(range(conf.NODES_COUNT))
	_data_dist_node_idx = conf.INDEX

match conf.DATA_DISTRIBUTION:
	case 'all':
		train_data, test_data = train_data_all, test_data_all
		
		if not hasattr(train_data, 'indices'):
			train_data.indices = list(range(len(train_data)))
		if not hasattr(test_data, 'indices'):
			test_data.indices = list(range(len(test_data)))
	case 'prepared-file':  # FIXME: rename to `prepared-file-indices`.
		_train_file, _test_file = Path(f'./run/data_dist/{conf.INDEX}-train.npy'), Path(f'./run/data_dist/{conf.INDEX}-test.npy')
		_train_split, _test_split = np.fromfile(_train_file, dtype=int), np.fromfile(_test_file, dtype=int)
		
		train_data, test_data = Subset(train_data_all, _train_split), Subset(test_data_all, _test_split)
		
		del _train_file, _test_file, _train_split, _test_split
	case 'iid':
		# TODO: the test proportions should be the same as the train proportions.
		train_splits = data_dist.iid(len_dataset=len(train_data_all), n=conf.DATA_DISTRIBUTION_NODES_COUNT, rng=Random(conf.GLOBAL_RANDOM_SEED))
		test_splits = data_dist.iid(len_dataset=len(test_data_all), n=conf.DATA_DISTRIBUTION_NODES_COUNT, rng=Random(conf.GLOBAL_RANDOM_SEED))
		
		train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
	case 'stratified':
		_rng = Random(conf.GLOBAL_RANDOM_SEED)
		
		# TODO: shuffle per label stratified data distribution.
		train_splits = data_dist.stratified(labels=train_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, rng=_rng)
		test_splits = data_dist.stratified(labels=test_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, rng=_rng)
		
		train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
		
		del _rng
	case 'one-ten':
		assert conf.DATA_DISTRIBUTION_NODES_COUNT == 10 and num_labels == 10
		
		train_splits = data_dist.by_labels(labels=train_data_all.targets, num_labels=num_labels)
		test_splits = data_dist.by_labels(labels=test_data_all.targets, num_labels=num_labels)
		
		train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
	case _:
		if conf.DATA_DISTRIBUTION == 'non-iid-strict-label-dir' or conf.DATA_DISTRIBUTION.startswith('non-iid-strict-label-dir-'):
			_alpha = 0.5 if conf.DATA_DISTRIBUTION == 'non-iid-strict-label-dir' else float(conf.DATA_DISTRIBUTION.split('-', 5)[-1])
			_np_rng = np.random.default_rng(conf.GLOBAL_RANDOM_SEED)
			
			train_splits, _labels_proportions = data_dist.non_iid_strict_label_dir(labels=train_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, alpha=_alpha, np_rng=_np_rng, return_labels_proportions=True)
			test_splits = data_dist.non_iid_strict_label_dir(labels=test_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, alpha=_alpha, np_rng=_np_rng, labels_proportions=_labels_proportions)
			
			train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
			
			del _alpha, _np_rng, _labels_proportions
		elif conf.DATA_DISTRIBUTION == 'non-iid-label-dir' or conf.DATA_DISTRIBUTION.startswith('non-iid-label-dir-'):
			if conf.DATA_DISTRIBUTION == 'non-iid-label-dir':
				_alpha = 0.5
				_min_required_size = 10
			else:
				_args = conf.DATA_DISTRIBUTION.removeprefix('non-iid-label-dir-').split('-')
				
				_alpha = float(_args[0])
				_min_required_size = int(_args[1]) if len(_args) > 1 else 10
			
			_np_rng = np.random.default_rng(conf.GLOBAL_RANDOM_SEED)
			
			train_splits, _labels_proportions = data_dist.non_iid_label_dir(labels=train_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, alpha=_alpha, np_rng=_np_rng, return_labels_proportions=True, min_required_size=_min_required_size)
			test_splits = data_dist.non_iid_label_dir(labels=test_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, alpha=_alpha, np_rng=_np_rng, labels_proportions=_labels_proportions, min_required_size=_min_required_size)
			
			train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
			
			del _alpha, _np_rng, _labels_proportions, _min_required_size
		elif conf.DATA_DISTRIBUTION.startswith('non-iid-'):
			_args = conf.DATA_DISTRIBUTION.removeprefix('non-iid-').split('-')
			_shards_factor = int(_args[0])
			_shuffle = (_args[1] in ('true', 'y', 'yes', '1')) if len(_args) >= 2 else False
			
			train_splits = data_dist.non_iid(labels=train_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, shards_factor=_shards_factor)
			test_splits = data_dist.non_iid(labels=test_data_all.targets, n=conf.DATA_DISTRIBUTION_NODES_COUNT, shards_factor=_shards_factor)
			
			if _shuffle:
				# There isn't an ordering guarantee in this method, unlike the `pretty-non-iid` method; therefore, shuffle the splits before assigning them.
				_rng = Random(conf.GLOBAL_RANDOM_SEED)
				_splits = list(zip(train_splits, test_splits))
				_rng.shuffle(_splits)
				train_splits, test_splits = zip(*_splits)
				train_splits, test_splits = list(train_splits), list(test_splits)
				
				del _rng, _splits
			
			train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
			
			del _shards_factor
		elif conf.DATA_DISTRIBUTION.startswith('fixed-'):
			# TODO: move into `data_dist` as a function.
			
			_vars = conf.DATA_DISTRIBUTION.split('-', 1)[-1].split(',')
			_vars = [var.split('=') for var in _vars]
			_vars = {t[0]: t[1] for t in _vars}
			_vars = {k: v.split('/') for k, v in _vars.items()}
			_vars = {k: (tuple(map(lambda i: int(i), v[0].split('-'))), int(v[1])) for k, v in _vars.items()}
			_dist = _vars
			del _vars
			
			_train_class_map = {str(i): torch.nonzero(torch.as_tensor(train_data_all.targets) == i, as_tuple=False).squeeze().tolist() for i in range(len(train_data_all.classes))}
			_test_class_map = {str(i): torch.nonzero(torch.as_tensor(test_data_all.targets) == i, as_tuple=False).squeeze().tolist() for i in range(len(test_data_all.classes))}
			
			_rnd = Random(conf.GLOBAL_RANDOM_SEED)
			for indices in list(_train_class_map.values()) + list(_test_class_map.values()):
				_rnd.shuffle(indices)
			
			_train_indices, _test_indices = [], []
			for label, ((my_quota_start, my_quota_end), total_quota) in _dist.items():
				assert my_quota_start >= 0 and my_quota_end <= total_quota
				assert label in _train_class_map and label in _test_class_map
				
				_train_splits = [split.tolist() for split in np.array_split(_train_class_map[label], total_quota)]
				_test_splits = [split.tolist() for split in np.array_split(_test_class_map[label], total_quota)]
				
				_my_train_splits = _train_splits[my_quota_start:my_quota_end]
				_my_test_splits = _test_splits[my_quota_start:my_quota_end]
				
				_my_train_indices = sum(_my_train_splits, start=[])
				_my_test_indices = sum(_my_test_splits, start=[])
				
				_train_indices.extend(_my_train_indices)
				_test_indices.extend(_my_test_indices)
			
			train_data, test_data = Subset(train_data_all, _train_indices), Subset(test_data_all, _test_indices)
		elif conf.DATA_DISTRIBUTION.startswith('pretty-non-iid-'):
			_groups_count = int(conf.DATA_DISTRIBUTION.split('-', 3)[-1])
			_rng = Random(conf.GLOBAL_RANDOM_SEED)
			
			_unique_labels = np.unique_values(list(train_data_all.class_to_idx.values()) + list(test_data_all.class_to_idx.values())).tolist()
			train_splits, test_splits = data_dist.pretty_non_iid(
				train_data_all.targets, test_data_all.targets,
				n=conf.DATA_DISTRIBUTION_NODES_COUNT, groups_count=_groups_count, rng=_rng, unique_labels=_unique_labels
			)
			
			train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
			
			del _groups_count, _rng, _unique_labels
		elif conf.DATA_DISTRIBUTION == 'iid-strict-dir-quantity' or conf.DATA_DISTRIBUTION.startswith('iid-strict-dir-quantity-'):
			# TODO: move into `data_dist` as a function.
			
			_alpha = 0.5 if conf.DATA_DISTRIBUTION == 'iid-strict-dir-quantity' else float(conf.DATA_DISTRIBUTION.split('-', 4)[-1])
			
			_np_rng = np.random.default_rng(conf.GLOBAL_RANDOM_SEED)
			
			_proportions = _np_rng.dirichlet(np.repeat(_alpha, conf.DATA_DISTRIBUTION_NODES_COUNT))
			
			_nodes_train_counts = _np_rng.multinomial(len(train_data_all), _proportions)
			_nodes_test_counts = _np_rng.multinomial(len(test_data_all), _proportions)
			
			_train_splits_points, _test_splits_points = np.cumsum(_nodes_train_counts)[:-1], np.cumsum(_nodes_test_counts)[:-1]
			
			_train_indices, _test_indices = _np_rng.permutation(len(train_data_all)), _np_rng.permutation(len(test_data_all))
			
			_nodes_train_indices = [a.tolist() for a in np.split(_train_indices, _train_splits_points)]
			_nodes_test_indices = [a.tolist() for a in np.split(_test_indices, _test_splits_points)]
			
			if conf.INDEX == 0:
				assert len(set(sum(_nodes_train_indices, start=[]))) == len(train_data_all)
				assert len(set(sum(_nodes_test_indices, start=[]))) == len(test_data_all)
			
			train_data, test_data = Subset(train_data_all, _nodes_train_indices[_data_dist_node_idx]), Subset(test_data_all, _nodes_test_indices[_data_dist_node_idx])
		elif conf.DATA_DISTRIBUTION == 'iid-dir-quantity' or conf.DATA_DISTRIBUTION.startswith('iid-dir-quantity-'):
			# TODO: move into `data_dist` as a function.
			
			_alpha = 0.5 if conf.DATA_DISTRIBUTION == 'iid-dir-quantity' else float(conf.DATA_DISTRIBUTION.split('-', 3)[-1])
			
			_np_rng = np.random.default_rng(conf.GLOBAL_RANDOM_SEED)
			
			_train_indices, _test_indices = _np_rng.permutation(len(train_data_all)), _np_rng.permutation(len(test_data_all))
			
			_min_size, _min_required_size = 0, 10
			_proportions = None
			while _min_size < _min_required_size:
				_proportions = _np_rng.dirichlet(np.repeat(_alpha, conf.DATA_DISTRIBUTION_NODES_COUNT))
				_proportions /= _proportions.sum()
				_min_size = np.min(_proportions * len(train_data_all))
			
			_train_splits_points = (np.cumsum(_proportions) * len(train_data_all)).astype(int)[:-1]
			_test_splits_points = (np.cumsum(_proportions) * len(test_data_all)).astype(int)[:-1]
			
			_nodes_train_indices = [a.tolist() for a in np.split(_train_indices, _train_splits_points)]
			_nodes_test_indices = [a.tolist() for a in np.split(_test_indices, _test_splits_points)]
			
			if conf.INDEX == 0:
				assert len(set(sum(_nodes_train_indices, start=[]))) == len(train_data_all)
				assert len(set(sum(_nodes_test_indices, start=[]))) == len(test_data_all)
			
			train_data, test_data = Subset(train_data_all, _nodes_train_indices[_data_dist_node_idx]), Subset(test_data_all, _nodes_test_indices[_data_dist_node_idx])
		elif conf.DATA_DISTRIBUTION.startswith('pathological-non-iid-'):
			_args = conf.DATA_DISTRIBUTION.removeprefix('pathological-non-iid-').split('-')
			_k = int(_args[0])
			
			_np_rng = np.random.default_rng(conf.GLOBAL_RANDOM_SEED)
			
			train_splits, test_splits = data_dist.pathological_non_iid(
				train_data_all.targets, test_data_all.targets,
				n=conf.DATA_DISTRIBUTION_NODES_COUNT, k=_k, np_rng=_np_rng,
				unique_labels=np.unique_values(list(train_data_all.class_to_idx.values()) + list(test_data_all.class_to_idx.values())).tolist(),
			)
			
			train_data, test_data = Subset(train_data_all, train_splits[_data_dist_node_idx]), Subset(test_data_all, test_splits[_data_dist_node_idx])
			
			del _args, _k
		else:
			assert False

# Log the labels' distribution of the train and test data partitions.

_unique_targets, _unique_targets_counts = np.unique([train_data_all.targets[i] for i in train_data.indices], return_counts=True)
_unique_targets, _unique_targets_counts = map(str, _unique_targets), map(int, _unique_targets_counts)
_targets_counts = dict(zip(_unique_targets, _unique_targets_counts))
_logger.info(f"My train targets' counts: {_targets_counts}.", extra={'type': 'data-distribution-train', 'counts': _targets_counts})
_logger.debug(f"Here are my first 5 indices of the train dataset: {train_data.indices[:min(5, len(train_data.indices))]}.")
del _unique_targets, _unique_targets_counts, _targets_counts  # TODO: move this into a function.

_unique_targets, _unique_targets_counts = np.unique([test_data_all.targets[i] for i in test_data.indices], return_counts=True)
_unique_targets, _unique_targets_counts = map(str, _unique_targets), map(int, _unique_targets_counts)
_targets_counts = dict(zip(_unique_targets, _unique_targets_counts))
_logger.info(f"My test targets' counts: {_targets_counts}.", extra={'type': 'data-distribution-test', 'counts': _targets_counts})
_logger.debug(f"Here are my first 5 indices of the test dataset: {test_data.indices[:min(5, len(test_data.indices))]}.")
del _unique_targets, _unique_targets_counts, _targets_counts  # TODO: move this into a function.

_pin_memory = conf.DATA_PIN_MEMORY and (conf_eval.DEVICE_EVAL.type != 'cpu')
_pin_memory_device = str(conf_eval.DEVICE_EVAL) if _pin_memory else ''

_logger.debug(f"Pin memory = {_pin_memory}; pin memory device = {_pin_memory_device}.")

_train_rng = torch.Generator()
_train_rng.manual_seed(conf.NODE_GLOBAL_RANDOM_SEED * conf.INDEX)
# noinspection PyUnboundLocalVariable
train_data_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, shuffle=True, pin_memory=_pin_memory, pin_memory_device=_pin_memory_device, collate_fn=_collate_fn if conf.DATA == 'femnist' else None, drop_last=conf.TRAIN_BATCH_DROP_LAST, generator=_train_rng)

# noinspection PyUnboundLocalVariable
train_eval_data_loader = DataLoader(train_data, batch_size=conf.BATCH_SIZE, shuffle=False, pin_memory=_pin_memory, pin_memory_device=_pin_memory_device, collate_fn=_collate_fn if conf.DATA == 'femnist' else None, drop_last=False)
# noinspection PyUnboundLocalVariable
test_data_loader = DataLoader(test_data, batch_size=conf.BATCH_SIZE, shuffle=False, pin_memory=_pin_memory, pin_memory_device=_pin_memory_device, collate_fn=_collate_fn if conf.DATA == 'femnist' else None, drop_last=False)

del _pin_memory, _pin_memory_device, _train_rng

_del_all = environ.get('DATA_DEL_ALL', 'true').lower() in {'true', '1', 't', 'y', 'yes'}

if _del_all:
	del train_data_all, test_data_all

del _del_all


def main():
	if conf.PREPARE_DATA_DISTRIBUTION:
		assert 'train_splits' in globals() and 'test_splits' in globals()
		
		data_dist_dir = Path('./run/data_dist')
		for node_idx in range(conf.NODES_COUNT):
			train_file = data_dist_dir / f'{node_idx}-train.npy'
			test_file = data_dist_dir / f'{node_idx}-test.npy'
			
			node_data_dist_idx = data_dist_nodes_indices[node_idx]
			
			if not train_file.exists():
				train_split = np.asarray(train_splits[node_data_dist_idx], dtype=int)
				train_split.tofile(train_file)
			else:
				print(f"File {train_file} exists, skipping.")
			
			if not test_file.exists():
				test_split = np.asarray(test_splits[node_data_dist_idx], dtype=int)
				test_split.tofile(test_file)
			else:
				print(f"File {test_file} exists, skipping.")


if __name__ == '__main__':
	main()

del _data_dist_node_idx

_del_splits = environ.get('DATA_DEL_SPLITS', 'True').lower() in {'true', '1', 't', 'y', 'yes'}

if _del_splits:
	if 'train_splits' in globals():
		del train_splits
	if 'test_splits' in globals():
		del test_splits
	
	del data_dist_nodes_indices

del _del_splits
