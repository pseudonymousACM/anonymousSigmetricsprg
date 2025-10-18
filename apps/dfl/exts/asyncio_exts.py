import asyncio
import functools
from typing import Callable, Any, override

# noinspection PyProtectedMember
from p2psims.extensions.queue import _ItemWithComparator


class AsyncComparatorOrderPriorityQueue(asyncio.PriorityQueue):
	def __init__(self, comparator: Callable[[Any, Any], int], maxsize=-1):
		super().__init__(maxsize)
		self._comparator = comparator
	
	@override
	def _get(self):
		return super()._get().item
	
	@override
	def _put(self, item):
		super()._put(_ItemWithComparator(item, self._comparator))


def once(f):
	lock = asyncio.Lock()
	
	ran = False
	res = None
	
	@functools.wraps(f)
	async def wrapper(*args, **kwargs):
		nonlocal ran, res
		
		if ran:
			return res
		
		async with lock:
			if ran:
				return res
			else:
				res = await f(*args, **kwargs)
				ran = True
				return res
	
	async def reset():
		nonlocal ran, res
		
		async with lock:
			ran = False
			res = None
	
	wrapper.reset = reset
	return wrapper


# TODO: allow `Callable[[], Awaitable[Any]]` as a `state_getter`.
def once_per_state(state_getter: Callable[[], Any]):
	lock = asyncio.Lock()
	
	last_state = None
	last_res = None
	
	def decorator(f):
		@functools.wraps(f)
		async def wrapper(*args, **kwargs):
			nonlocal last_state, last_res
			
			curr_state = state_getter()
			if curr_state == last_state:
				return last_res
			
			async with lock:
				if curr_state == last_state:
					return last_res
				else:
					last_res = await f(*args, **kwargs)
					last_state = curr_state
					
					return last_res
		
		async def reset():
			nonlocal last_state, last_res
			
			async with lock:
				last_state = None
				last_res = None
		
		wrapper.reset = reset
		return wrapper
	
	return decorator
