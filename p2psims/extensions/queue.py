import functools
from queue import PriorityQueue
from typing import Any, Callable, override


class ComparatorOrderPriorityQueue(PriorityQueue):
	def __init__(self, comparator: Callable[[Any, Any], int], maxsize=-1):
		super().__init__(maxsize)
		self._comparator = comparator
	
	@override
	def _get(self):
		return super()._get().item
	
	@override
	def _put(self, item):
		super()._put(_ItemWithComparator(item, self._comparator))


@functools.total_ordering
class _ItemWithComparator:
	def __init__(self, item, comparator: Callable[[Any, Any], int]):
		self.item = item
		self._comparator = comparator
	
	def __lt__(self, other):
		return self._comparator(self.item, other.item) < 0
	
	def __gt__(self, other):
		return self._comparator(self.item, other.item) > 0
	
	def __le__(self, other):
		return self._comparator(self.item, other.item) <= 0
	
	def __ge__(self, other):
		return self._comparator(self.item, other.item) >= 0
	
	def __eq__(self, other):
		return self._comparator(self.item, other.item) == 0


def compare_objects(a, b) -> int:
	if a > b:
		return 1
	elif a < b:
		return -1
	else:  # Assuming total order (implicit a == b).
		return 0
