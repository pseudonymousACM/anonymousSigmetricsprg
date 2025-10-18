import logging
from typing import override


class LoggerAdapter(logging.LoggerAdapter):
	def __init__(self, logger, extra=None, overwrite: bool = False):
		super().__init__(logger, extra)
		self.overwrite = overwrite
	
	@override
	def process(self, msg, kwargs):
		if 'extra' not in kwargs:
			kwargs['extra'] = {}
		else:
			kwargs['extra'] = kwargs['extra'].copy()
		
		if self.overwrite:
			kwargs['extra'] |= self.extra
		else:
			extra = kwargs['extra']
			for k, v in self.extra.items():
				if k not in extra:
					extra[k] = v
		
		return msg, kwargs
