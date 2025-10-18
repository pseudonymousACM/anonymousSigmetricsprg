import numpy as np
import torch
from torch_lr_finder import LRFinder, TrainDataLoaderIter


def find_lr_steep_grad(lr_finder: LRFinder) -> tuple[float, float]:
	losses, lrs = lr_finder.history['loss'], lr_finder.history['lr']
	min_grad_idx = np.gradient(losses).argmin()
	return lrs[min_grad_idx], losses[min_grad_idx]


def find_max_lr_flat_loss_lte(lr_finder: LRFinder, tol: float, lr_lte: float | None = None) -> float:
	lrs, losses = lr_finder.history['lr'], lr_finder.history['loss']
	if lr_lte is not None:
		cut_idx = None
		for i, lr in reversed(list(enumerate(lrs))):
			if lr <= lr_lte:
				cut_idx = i
				break
		else:
			raise ValueError(f"Found no LR LTE to the given `lr_lte={lr_lte}`.")
		
		if cut_idx is not None:
			lrs, losses = lrs[:cut_idx], losses[:cut_idx]
	
	abs_grads = np.abs(np.gradient(losses))
	indices = np.where(abs_grads <= tol)[0]
	return lrs[indices[-1] if len(indices) > 0 else 0]


def find_min_lr_flat_loss_gte(lr_finder: LRFinder, tol: float, lr_gte: float | None = None) -> float:
	losses, lrs = lr_finder.history['loss'], lr_finder.history['lr']
	if lr_gte is not None:
		cut_idx = None
		for i, lr in enumerate(lrs):
			if lr >= lr_gte:
				cut_idx = i
				break
		else:
			raise ValueError(f"Found no LR GTE to the given `lr_gte={lr_gte}`.")
		
		if cut_idx is not None:
			lrs, losses = lrs[cut_idx:], losses[cut_idx:]
	
	abs_grads = np.abs(np.gradient(losses))
	indices = np.where(abs_grads <= tol)[0]
	return lrs[indices[0] if len(indices) > 0 else -1]


class MyLrFinder(LRFinder):
	# noinspection PyAttributeOutsideInit
	def range_test(self, train_loader, val_loader=None, start_lr=None, end_lr=10, num_iter=100, step_mode="exp", smooth_f=0.05, diverge_th=5, accumulation_steps=1, non_blocking_transfer=True, epochs_per_iter: int | None = None, reset_per_iter: bool = False):
		try:
			self._original_train_loader = train_loader
			self._epochs_per_iter = epochs_per_iter
			self._reset_per_iter = reset_per_iter
			
			super().range_test(train_loader, val_loader, start_lr, end_lr, num_iter, step_mode, smooth_f, diverge_th, accumulation_steps, non_blocking_transfer)
		finally:
			del self._original_train_loader
			del self._epochs_per_iter
			del self._reset_per_iter
	
	def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
		if self._epochs_per_iter is None:
			res = super()._train_batch(train_iter, accumulation_steps, non_blocking_transfer)
			
			if self._reset_per_iter:
				self.reset()
			
			return res
		else:
			assert self._epochs_per_iter > 0
			assert accumulation_steps <= 1
			
			train_iter = self._original_train_loader
			assert not isinstance(train_iter, TrainDataLoaderIter)
			
			running_losses_sum, running_total = 0, 0
			
			self.model.train()
			for _ in range(self._epochs_per_iter):
				for inputs, labels in train_iter:
					inputs, labels = self._move_to_device(
						inputs, labels, non_blocking=non_blocking_transfer
					)
					
					self.optimizer.zero_grad()
					
					# Forward pass.
					if self.amp_backend == "torch":
						with torch.amp.autocast(**self.amp_config):
							outputs = self.model(inputs)
							loss = self.criterion(outputs, labels)
					else:
						outputs = self.model(inputs)
						loss = self.criterion(outputs, labels)
					
					running_losses_sum += loss.item() * len(labels)
					running_total += len(labels)
					
					# Backward pass.
					if self.amp_backend == "torch":
						self.grad_scaler.scale(loss).backward()
					elif self.amp_backend == "apex" and hasattr(self.optimizer, "_amp_stash"):
						# noinspection PyUnresolvedReferences
						with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
							scaled_loss.backward()
					else:
						loss.backward()
					
					if self.amp_backend == "torch":
						self.grad_scaler.step(self.optimizer)
						self.grad_scaler.update()
					else:
						self.optimizer.step()
			
			if self._reset_per_iter:
				self.reset()
			
			return running_losses_sum / running_total
	
	# TODO: added `running_total`; contribute this change to the upstream; this makes the loss mean computation more accurate when having `drop_last=True` in the data loader.
	def _validate(self, val_iter, non_blocking_transfer=True):
		# Set model to evaluation mode and disable gradient computation.
		running_loss = 0
		running_total = 0
		self.model.eval()
		with torch.no_grad():
			for inputs, labels in val_iter:
				# Move data to the correct device.
				inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)
				
				# Forward pass and loss computation.
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				running_loss += loss.item() * len(labels)
				running_total += len(labels)
		
		return running_loss / running_total
