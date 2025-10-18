from typing import Literal

import plotly.express as px
import plotly.graph_objs as go

_ERROR_MODES = {'bar', 'band', 'bars', 'bands', None}


def line(*args, error_y_mode: Literal['bar', 'band', 'bars', 'bands', None] = None, error_y_band_alpha: float = 0.3, **kwargs):
	"""
	Extension of `plotly.express.line` to use error bands.
	
	Adapted from https://stackoverflow.com/a/69594497/6055075. TODO: contribute the fixes (the adaptations) to the upstream/source.
	"""
	
	if error_y_mode in {'bar', 'bars', None}:
		return px.line(*args, **kwargs)
	elif error_y_mode in {'band', 'bands'}:
		if 'error_y' not in kwargs:
			raise ValueError(f"If you provide argument 'error_y_mode' you must also provide 'error_y'.")
		
		figure_with_error_bars = px.line(*args, **kwargs)
		fig = px.line(*args, **{arg: val for arg, val in kwargs.items() if arg != 'error_y'})
		
		for data in figure_with_error_bars.data:
			x = list(data['x'])
			
			y_upper = list(data['y'] + data['error_y']['array'])
			y_lower = list(data['y'] - data['error_y']['array'] if data['error_y']['arrayminus'] is None else data['y'] - data['error_y']['arrayminus'])
			
			color = f"rgba({tuple(int(data['line']['color'].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))},{error_y_band_alpha})".replace('((', '(').replace('),', ',').replace(' ', '')
			
			fig.add_trace(
				go.Scatter(
					x=x + x[::-1], y=y_upper + y_lower[::-1],
					fill='toself', fillcolor=color,
					line=dict(color='rgba(255,255,255,0)'),  # Removes the error-bands' border-lines.
					hoverinfo="skip", showlegend=False,
					legendgroup=data['legendgroup'],
					xaxis=data['xaxis'], yaxis=data['yaxis'],
				)
			)
		
		# Reorder data as said here: https://stackoverflow.com/a/66854398/8849755.
		reordered_data = []
		for i in range(int(len(fig.data) / 2)):
			reordered_data.append(fig.data[i + int(len(fig.data) / 2)])
			reordered_data.append(fig.data[i])
		fig.data = tuple(reordered_data)
		
		return fig
	else:
		raise ValueError(f"'error_y_mode' must be one of {_ERROR_MODES}, received {repr(error_y_mode)}.")
