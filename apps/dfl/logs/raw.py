from datetime import timedelta
from pathlib import Path
from typing import Any

from jsonl import Jsonl


def accs(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			type_ = log.get('type', None)
			if type_ in ('train-accuracy', 'test-accuracy'):
				raw.append({
					'type': type_.removesuffix('-accuracy'),
					'file': file.name,
					'round': log['round'],
					'accuracy': log['accuracy'],
					'mean_loss': log['mean-loss'],
				})
			elif type_ == 'test-accuracy-pre-agg':
				raw.append({
					'type': 'test-pre-agg',
					'file': file.name,
					'round': log['round'],
					'accuracy': log['accuracy'],
					'mean_loss': log['mean-loss'],
				})
			elif type_ == 'raw-train-accuracy':
				acc_type = type_.removesuffix('-accuracy')
				for epoch in range(log['epochs']):
					acc, loss = log['accuracies'][epoch], log['losses'][epoch]
					raw.append({
						'type': acc_type,
						'file': file.name,
						'round': log['round'] + (epoch / log['epochs']),
						'accuracy': acc,
						'mean_loss': loss
					})
			else:
				pass  # NOP.
	
	return raw


def durs(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			match log.get('type', None):
				case 'time-pull' | 'time-train' | 'time-search-ready' | 'time-round':
					raw.append({
						'type': log['type'].split('-', maxsplit=1)[1],
						'file': file.name,
						'round': log['round'],
						'duration': timedelta(seconds=log['time-seconds']),
					})
				case 'search':
					raw.append({
						'type': 'search',
						'file': file.name,
						'round': log['round'],
						'duration': timedelta(seconds=log['time-seconds']),
					})
				case _:
					pass  # NOP.
	
	return raw


def searches(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('search',):
				d = {
					'file': file.name,
					'round': log['round'],
					'min_similarity': log['min_similarity'],
					'k': log.get('k', 0),
					'sequential': log.get('sequential', False),
					'duration': timedelta(seconds=log['time-seconds']),
					'result': log['result'],
				}
				
				if 'target_k' in log:
					d['target_k'] = log['target_k']
				if 'isotonic_xys' in log:
					d['isotonic_xys'] = log['isotonic_xys']
				
				raw.append(d)
			else:
				pass  # NOP.
	
	return raw


def data_dist(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('data-distribution-train', 'data-distribution-test'):
				raw.append({
					'file': file.name,
					'type': log['type'].split('-')[-1],
					'counts': log['counts'],
				})
			else:
				pass  # NOP.
	
	return raw


def agg_peers(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('agg-peers',):
				d = {
					'file': file.name,
					'round': log['round'],
					'peers': log['peers'],
				}
				
				if 'node' in log:
					d['node'] = log['node']
				
				raw.append(d)
			else:
				pass  # NOP.
	
	return raw


def init_peers(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('initial-peers',):
				raw.append({
					'file': file.name,
					'peers': log['peers'],
				})
			else:
				pass  # NOP.
	
	return raw


def zones(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('zone',):
				# TODO: also consider `join-zone` and `zone-initiator` log types; validate against them to ensure there is no inconsistency.
				
				raw.append({
					'file': file.name,
					'peers': {peer.rsplit(':', maxsplit=1)[0] for peer in log['peers']},
					'round': log['round'],
					'level': log['level'],
				})
			else:
				pass  # NOP.
	
	return raw


def clusters(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('cluster',):
				raw.append({
					'file': file.name,
					'round': log['round'],
					'level': log['level'],
					'peers': {peer.rsplit(':', maxsplit=1)[0] for peer in log['peers']},
				})
			else:
				pass  # NOP.
	
	return raw


def super_clusters_representatives(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('super-cluster-representative',):
				raw.append({
					'file': file.name,
					'round': log['round'],
					'level': log['level'],
					'super_clusters_representatives': {addr.rsplit(':', maxsplit=1)[0] for addr in log['super-clusters-representatives']},
				})
			else:
				pass  # NOP.
	
	return raw


def sims(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('similarities',):
				raw.append({
					'file': file.name,
					'round': log['round'],
					'sims': {frozenset((pa, pb)): sim for (pa, pb), sim in log['similarities']},
				})
			else:
				pass  # NOP.
	
	return raw


def son_sims(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('son-sims',):
				raw.append({
					'file': file.name,
					'round': log['round'],
					'sims': {tuple((pa, pb)): sim for (pa, pb), sim in log['sims']},
				})
			else:
				pass  # NOP.
	
	return raw


def sim_lats(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('simulate-latency',):
				raw.append({
					'file': file.name,
					'peer': log['peer'],
					'latency_ms': log['latency-ms'],
					'tc_id': log['tc_id'],
					'ip': log['ip'],
				})
			else:
				pass  # NOP.
	
	return raw


def confs(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('env',):
				raw.append({
					'file': file.name,
					'env': log['env'],
				})
			else:
				pass  # NOP.
	
	return raw


def models(run_dir: Path) -> list[dict[str, Any]]:
	raw = []
	
	for file in run_dir.iterdir():
		if not file.is_file() or file.suffix != ".jsonl":
			continue
		
		for log in Jsonl(file):
			if log.get('type', None) in ('model',):
				raw.append({
					'file': file.name,
					'init_fingerprint': log['init_fingerprint']
				})
			else:
				pass  # NOP.
	
	return raw
