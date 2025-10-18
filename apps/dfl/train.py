import asyncio

import conf
import dfl
import dfl.extensions.asyncio
import dfl.extensions.torch as torch_exts
import singletons
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


async def _train_epoch(model, optimizer, criterion, data_loader, device=None, epilogue_zero_grad: bool = True) -> tuple[int, int, float]:
	running_corrects, running_total, running_losses_sum = 0, 0, 0
	async for batch, target in dfl.extensions.asyncio.iter_to_aiter(iter(data_loader)):
		corrects, total, loss = await asyncio.to_thread(torch_exts.train_batch, model, optimizer, criterion, batch, target, device=device)
		
		running_corrects += corrects
		running_total += total
		running_losses_sum += loss * total
		
		if conf.GRADS_BASED_SEARCH_DATA:
			# TODO: would it be okay if the batches' sizes differ, or should we weigh each batch based on its size?
			
			if singletons.round_train_accumulated_params_grads is None:
				singletons.round_train_accumulated_params_grads = [param.grad.detach().clone() for param in model.parameters()]
			else:
				for i, param in enumerate(model.parameters()):
					singletons.round_train_accumulated_params_grads[i].add_(param.grad.detach())
	
	if epilogue_zero_grad:
		optimizer.zero_grad()
	
	return running_corrects, running_total, running_losses_sum


async def _train(
		model: nn.Module,
		optimizer: Optimizer,
		criterion,
		data_loader: DataLoader,
		epochs: int | None = None,
		device=None,
		epilogue_zero_grad: bool = True,
) -> list[tuple[int, int, float]]:
	model.train()
	
	epochs_metrics = []
	done_epochs = 0
	while epochs is None or done_epochs < epochs:
		metrics = await _train_epoch(model, optimizer, criterion, data_loader, device=device, epilogue_zero_grad=False)
		epochs_metrics.append(metrics)
		
		done_epochs += 1
	
	if epilogue_zero_grad:
		optimizer.zero_grad()
	
	return epochs_metrics


class TorchContext(torch_exts.Context):
	async def train(self, epochs: int | None = None):
		return await _train(self.model, self.optimizer, self.criterion, self.train_data_loader, epochs, device=self.device)
