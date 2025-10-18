import math


def find(data_quantities: list[int], target_avg_updates_per_epoch: int, pre_avg: bool = True, drop_last: bool | None = None) -> int:
	if pre_avg:
		sum_data = sum(data_quantities)
		avg_data = sum_data / len(data_quantities)
		
		bs = avg_data / target_avg_updates_per_epoch
		if bs <= 0:
			return 1
		else:
			return 2 ** math.floor(math.log2(bs))
	else:
		for bs in (2 ** k for k in range(math.ceil(math.log2(max(data_quantities))), 0, -1)):
			if avg_updates_per_epoch(data_quantities, bs, drop_last=drop_last) >= target_avg_updates_per_epoch:
				return bs
		
		return 1


def avg_updates_per_epoch(data_quantities: list[int], batch_size: int, drop_last: bool | None = None) -> float:
	return sum(updates_per_epoch(data_quantities, batch_size, drop_last=drop_last)) / len(data_quantities)


def updates_per_epoch(data_quantities: list[int], batch_size: int, drop_last: bool | None = None) -> list[int | float]:
	return [
		(dq / batch_size) if drop_last is None else math.floor(dq / batch_size) if drop_last else math.ceil(dq / batch_size)
		for dq in data_quantities
	]
