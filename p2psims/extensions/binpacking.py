from typing import Any, Callable

from .queue import ComparatorOrderPriorityQueue, compare_objects


class _Bin:
	def __init__(self) -> None:
		self.weights_sum: float = 0
		self.items = list[Any]()
	
	def add_item(self, item: Any, weight: float) -> None:
		self.items.append(item)
		self.weights_sum += weight


# noinspection PyProtectedMember
def to_constant_bin_number_with_max_items_per_bin_and_fair_volume(items: list[Any], bins_count: int, max_items_per_bin: float, weight_key: Callable[[Any], Any] | None = None) -> list[list[Any]]:
	bins = [_Bin() for _ in range(bins_count)]
	pq = ComparatorOrderPriorityQueue(lambda a, b: compare_objects((a.weights_sum, len(a.items)), (b.weights_sum, len(b.items))))
	for bin_ in bins:
		pq._put(bin_)
	
	for item in sorted(items, key=weight_key, reverse=True):
		while True:
			current_bin = pq._get()
			if len(current_bin.items) < max_items_per_bin:
				current_bin.add_item(item, weight_key(item) if weight_key is not None else item)
				pq._put(current_bin)
				break
			else:
				pass
	
	return [bin_.items for bin_ in bins]
