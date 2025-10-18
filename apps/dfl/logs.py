import logging
from datetime import datetime
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

import conf

_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.DEBUG if conf.STDERR_DEBUG_LOG else logging.INFO)

_log_file_path = Path(f"./logs/{conf.NAME} {datetime.now().astimezone()}.jsonl")  # TODO: this file name is not Windows-safe.
_file_handler = logging.FileHandler(_log_file_path)
_file_handler.setFormatter(JsonFormatter("{asctime}{name}", style='{', rename_fields={'asctime': 'time'}))
_file_handler.setLevel(logging.INFO)

logging.basicConfig(
	format="{name}\t{asctime}\t{levelname}\t{message}\t",
	style='{',
	handlers=[_stderr_handler, _file_handler],
	level=logging.DEBUG
)

logging.getLogger('PIL.PngImagePlugin').disabled = True
logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
