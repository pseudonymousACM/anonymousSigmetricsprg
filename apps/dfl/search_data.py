import asyncio
import functools
import logging

import torch
from torch import nn

import conf
import conf_eval
import similarity
import singletons
from exts.asyncio_exts import once_per_state
from exts.torch_exts import topp_mag_tensors
from model import model

_logger = logging.getLogger(__name__)


# noinspection PyShadowingNames
def _list_vectorize(model: nn.Module | dict[str, torch.Tensor], grad: bool = False) -> list[torch.Tensor]:
	if isinstance(model, nn.Module):
		model = model.state_dict(keep_vars=True)
	elif not isinstance(model, dict):
		raise ValueError(f"Unknown model type {type(model)}.")
	
	model = sorted(model.items(), key=lambda kv: kv[0])
	
	if not grad:
		return [param.detach().flatten() for _, param in model]
	else:
		r = [param.grad.detach().flatten() for _, param in model if param.grad is not None]
		
		if len(r) == 0:
			raise ValueError(f"No gradients (or parameters, hypothetically).")
		
		return r


# TODO: remove the `cpu` argument in this functions if it is just a tail operation.

# noinspection PyShadowingNames
def params(model: nn.Module | dict[str, torch.Tensor], cpu: bool = True, cat: bool = True) -> torch.Tensor | list[torch.Tensor]:
	res = _list_vectorize(model)
	if cat:
		res = torch.cat(res)
		if cpu:
			res = res.cpu()
	else:
		if cpu:
			res = [r.cpu() for r in res]
	
	return res


# noinspection PyShadowingNames
def grads(model: nn.Module | dict[str, torch.Tensor], cpu: bool = True, cat: bool = True) -> torch.Tensor | list[torch.Tensor]:
	res = _list_vectorize(model, grad=True)
	if cat:
		res = torch.cat(res)
		if cpu:
			res = res.cpu()
	else:
		if cpu:
			res = [r.cpu() for r in res]
	
	return res


# noinspection PyShadowingNames
def params_top_p(model: nn.Module | dict[str, torch.Tensor], p: float, cpu: bool = True) -> torch.Tensor:
	model_params = params(model, cpu=False, cat=False)
	model_params_topp = topp_mag_tensors(model_params, p)
	res = torch.cat(model_params_topp)
	
	if cpu:
		res = res.cpu()
	return res


# noinspection PyShadowingNames
def grads_top_p(model: nn.Module | dict[str, torch.Tensor], p: float, cpu: bool = True, cat: bool = True) -> torch.Tensor | list[torch.Tensor]:
	model_grads = grads(model, cpu=False, cat=False)
	model_grads_topp = topp_mag_tensors(model_grads, p)
	
	if cat:
		res = torch.cat(model_grads_topp)
		if cpu:
			res = res.cpu()
	else:
		res = model_grads_topp
		if cpu:
			res = [r.cpu() for r in res]
	
	return res


# noinspection PyShadowingNames
def params_grads_top_p(model: nn.Module | dict[str, torch.Tensor], p: float, cpu: bool = True) -> torch.Tensor:
	model_grads_topp = grads_top_p(model, p, cpu=False, cat=False)
	model_params = params(model, cpu=False, cat=False)
	
	model_params_topp = []
	for model_param, model_grad_topp in zip(model_params, model_grads_topp):
		model_grad_topp.values().copy_(model_param[model_grad_topp.indices().squeeze(dim=0)])
		
		model_param_topp = model_grad_topp
		model_params_topp.append(model_param_topp)
	
	res = torch.cat(model_params_topp)
	if cpu:
		res = res.cpu()
	
	return res


match conf.SEARCH_DATA:
	case 'params':
		_sync_build = params
	case 'grads':
		_sync_build = grads
	case _:
		if conf.SEARCH_DATA.startswith('params-top-'):
			_p = float(conf.SEARCH_DATA.split('-', 2)[-1])
			_sync_build = functools.partial(params_top_p, p=_p)
		elif conf.SEARCH_DATA.startswith('grads-top-'):
			_p = float(conf.SEARCH_DATA.split('-', 2)[-1])
			_sync_build = functools.partial(grads_top_p, p=_p)
		elif conf.SEARCH_DATA.startswith('params-grads-top-'):
			_p = float(conf.SEARCH_DATA.split('-', 3)[-1])
			_sync_build = functools.partial(params_grads_top_p, p=_p)
		else:
			assert False


# noinspection PyShadowingNames
async def build(model: nn.Module | dict[str, torch.Tensor], cpu: bool = True) -> torch.Tensor:
	# noinspection PyTypeChecker
	return await asyncio.to_thread(_sync_build, model, cpu=cpu)


def _apply_round_train_acuumulated_grads():
	global model
	
	for i, param in enumerate(model.parameters()):
		param.grad = singletons.round_train_accumulated_params_grads[i]


@once_per_state(lambda: singletons.round_)
async def _get() -> torch.Tensor:
	if conf.GRADS_BASED_SEARCH_DATA:
		_apply_round_train_acuumulated_grads()
	
	if conf_eval.SON_DATA_DEVICE_EVAL.type != 'cpu':
		ret = await build(model, cpu=False)
		ret = await asyncio.to_thread(ret.to, conf_eval.SON_DATA_DEVICE_EVAL, non_blocking=True)
		return ret
	else:
		return await build(model, cpu=True)


async def reset() -> None:
	await _get.reset()
	
	if conf.GRADS_BASED_SEARCH_DATA:
		singletons.round_train_accumulated_params_grads = None


@once_per_state(lambda: singletons.round_)
async def publish() -> None:
	round_ = singletons.round_
	data = await _get()
	
	meta = {'from': conf.NAME, 'round': round_, 'search_data': True}
	await singletons.dfl_communicator.publish(meta, data, pre_encode=True)
	if conf.SEARCH_FUL_PEERS_SELECTION and round_ >= 0 and round_ % conf.SEARCH_RENEW_ROUNDS == 0:  # SON renewal round.
		# Always keep the SON search data published until it is obsolete; it is useful for example for the `par-topk-ideal-th-k=7` peers selection method;
		# no extra memory is used as the SON data is kept in the memory anyway.
		last_round = round_ + conf.SEARCH_RENEW_ROUNDS - 1
		await singletons.dfl_node.register_round_cleanup(lambda: singletons.dfl_communicator.unpublish(meta), round_=last_round)
	else:
		await singletons.dfl_node.register_round_cleanup(lambda: singletons.dfl_communicator.unpublish(meta), round_=round_)
	
	logging.debug(f"Published {meta}.")


# TODO: change the default of `force_comm` to `True`.
async def get(node: str | None = None, round_: int | None = None, last_son_renewal_round: bool = False, force_comm: bool = False) -> torch.Tensor:
	if not force_comm and (not last_son_renewal_round and (node in (None, conf.NAME)) and (round_ in (None, singletons.round_))):
		return await _get()
	else:
		if node is None:
			node = conf.NAME
		if round_ is None:
			round_ = singletons.round_
		
		if last_son_renewal_round:
			round_ -= (round_ % conf.SEARCH_RENEW_ROUNDS)
		
		meta = {'from': node, 'round': round_, 'search_data': True}
		
		if conf.CACHE_SEARCH_DATA:
			_, node_datum = await singletons.dfl_caching_communicator.subscribe(meta)
			if conf.CACHE_SON_SEARCH_DATA and round_ % conf.SEARCH_RENEW_ROUNDS == 0:  # SON renewal round.
				last_round = round_ + conf.SEARCH_RENEW_ROUNDS - 1
				await singletons.dfl_node.register_round_cleanup(lambda: singletons.dfl_caching_communicator.uncache(meta, key_error=False), round_=last_round)  # FIXME: this registers many (`min(conf.SON_RENEWAL_ROUND`, `conf.ROUNDS`)) redundant/same cleanups for the same `last_round`, because it is called many times.
			else:
				await singletons.dfl_node.register_round_cleanup(lambda: singletons.dfl_caching_communicator.uncache(meta, key_error=False))
		else:
			_, node_datum = await singletons.dfl_communicator.subscribe(meta)
		
		# A last resort to move into the designated device; normally, it should already be there as it is saved/loaded with its device tag.
		node_datum = await asyncio.to_thread(node_datum.to, conf_eval.SON_DATA_DEVICE_EVAL, non_blocking=True)
		
		return node_datum


# TODO: change the default of `force_comm` to `True`.
# TODO: can cache the return values locally to share between `MONITOR_SIMILARITIES=n0` and `PEERS_SELECTION=ideal-top-k`.
async def sim(n0: str, n1: str | None = None, round_: int | None = None, last_son_renewal_round: bool | None = None, n1_datum: torch.Tensor | None = None, n0_last_son_renewal_round: bool | None = None, n1_last_son_renewal_round: bool | None = None, force_comm: bool = False) -> float:
	if round_ is None:
		round_ = singletons.round_
	if n1 is None:
		n1 = conf.NAME
	
	if last_son_renewal_round is None:
		last_son_renewal_round = False
	if n0_last_son_renewal_round is None:
		n0_last_son_renewal_round = last_son_renewal_round
	if n1_last_son_renewal_round is None:
		n1_last_son_renewal_round = last_son_renewal_round
	
	async with asyncio.TaskGroup() as tg:
		n0_datum = tg.create_task(get(node=n0, round_=round_, last_son_renewal_round=n0_last_son_renewal_round, force_comm=force_comm))
		if n1_datum is None:
			n1_datum = tg.create_task(get(node=n1, round_=round_, last_son_renewal_round=n1_last_son_renewal_round, force_comm=force_comm))
			n1_datum = await n1_datum  # Last task; OK to await here.
	
	n0_datum = await n0_datum
	
	return await similarity.acalc(n0_datum, n1_datum)
