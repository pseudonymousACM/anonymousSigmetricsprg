import functools
import hashlib
import logging

import data
import torch
import torch.nn as nn
import torchvision

import conf
import conf_eval
import repro


def one_nn(datum_size: torch.Size, num_labels: int) -> nn.Module:
	# noinspection PyShadowingNames
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(datum_size.numel(), num_labels),
	)
	
	return model


# Referred to as "MNIST 2NN" in McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
def two_nn(datum_size: torch.Size, num_labels: int) -> nn.Module:
	# noinspection PyShadowingNames
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(datum_size.numel(), 200),
		nn.ReLU(),
		nn.Linear(200, 200),
		nn.ReLU(),
		nn.Linear(200, num_labels)
	)
	
	return model


# Referred to as "MNIST CNN" in McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.
def cnn(datum_size: torch.Size, num_labels: int, lin_size: int) -> nn.Module:
	# noinspection PyShadowingNames
	model = nn.Sequential(
		nn.Conv2d(datum_size[0], 32, kernel_size=5, padding=2),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2),
		nn.Conv2d(32, 64, kernel_size=5, padding=2),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2),
		nn.Flatten(),
		nn.Linear(lin_size, 512),
		nn.ReLU(),
		nn.Linear(512, num_labels)
	)
	
	return model


def resnet18(datum_size: torch.Size, num_labels: int) -> torchvision.models.ResNet:
	# noinspection PyShadowingNames
	model = torchvision.models.resnet18(num_classes=num_labels)
	if datum_size[0] != model.conv1.in_channels:
		model.conv1 = nn.Conv2d(datum_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
	return model


_datum_size = data.train_data[0][(0 if conf.DATA != 'femnist' else 'image')].size()

match conf.MODEL:
	case '1nn':
		init_model = functools.partial(one_nn, datum_size=_datum_size, num_labels=data.num_labels)
	case '2nn':
		init_model = functools.partial(two_nn, datum_size=_datum_size, num_labels=data.num_labels)
	case 'cnn':
		match conf.DATA:
			case 'mnist' | 'emnist' | 'fmnist' | 'femnist':
				_lin_size = 64 * 7 * 7
			case 'cifar10' | 'cifar100':
				_lin_size = 64 * 8 * 8
			case _:
				if conf.DATA.startswith('emnist-'):
					_lin_size = 64 * 7 * 7
				else:
					assert False
		
		init_model = functools.partial(cnn, datum_size=_datum_size, num_labels=data.num_labels, lin_size=_lin_size)
	case 'resnet18':
		init_model = functools.partial(resnet18, datum_size=_datum_size, num_labels=data.num_labels)
	case _:
		assert False

del _datum_size

if conf.HOMO_MODEL_INIT:
	with repro.seed(conf.GLOBAL_RANDOM_SEED):
		model = init_model()
else:
	# noinspection PyRedeclaration
	model = init_model()

model = model.to(conf_eval.DEVICE_EVAL)


# noinspection PyShadowingNames
def _model_fingerprint(model: nn.Module) -> str:
	hash_gen = hashlib.md5()
	for _, param in model.state_dict().items():
		hash_gen.update(param.detach().cpu().numpy().tobytes())
	return hash_gen.hexdigest()


_model_init_fingerprint = _model_fingerprint(model)
logging.info(f"Model initialization fingerprint: {_model_init_fingerprint}.", extra={'type': 'model', 'init_fingerprint': _model_init_fingerprint})
del _model_init_fingerprint

if conf.INDEX == 0:
	logging.debug(f"My model has {sum(torch.numel(p) for p in model.parameters())} learnable parameters.")
