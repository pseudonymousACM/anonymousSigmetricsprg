from torch.optim import Adam, SGD

import conf
from model import model


def adam():
	lr = conf.LR if conf.LR is not None else 1e-3
	l2 = conf.L2 if conf.L2 is not None else 0
	return Adam(model.parameters(), lr=lr, weight_decay=l2)


def sgd():
	lr = conf.LR if conf.LR is not None else 1e-3
	l2 = conf.L2 if conf.L2 is not None else 0
	return SGD(model.parameters(), lr=lr, weight_decay=l2)


match conf.OPTIMIZER:
	case 'adam':
		init = adam
	case 'sgd':
		init = sgd
	case _:
		assert False
