import math
from collections import OrderedDict
from functools import reduce
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sklearn.metrics import ndcg_score

import raw


def add_node_from_file(df: pd.DataFrame, node_column_name: str = 'node') -> None:
	df[node_column_name] = df['file'].str.split().str[0]


def add_node_index_from_node(df: pd.DataFrame, node_column_name: str = 'node', node_index_column_name: str = 'node_index') -> None:
	df[node_index_column_name] = pd.to_numeric(df[node_column_name].str.extract(r'(\d+)')[0], errors='coerce').astype('Int32')


def confs(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.confs(raw_)
	
	df = pd.DataFrame(raw_)
	add_node_from_file(df)
	add_node_index_from_node(df)
	
	df = df.convert_dtypes()
	df.sort_values(['node_index'], inplace=True)
	df.reset_index(inplace=True)
	
	return df


def models(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.models(raw_)
	
	df = pd.DataFrame(raw_)
	add_node_from_file(df)
	add_node_index_from_node(df)
	
	df = df.convert_dtypes()
	df.sort_values(['node_index'], inplace=True)
	df.reset_index(inplace=True)
	
	return df


def init_peers(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.init_peers(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df)
	add_node_index_from_node(df)
	
	df = df.explode('peers', ignore_index=True)
	df.rename(columns={'peers': 'peer'}, inplace=True)
	add_node_index_from_node(df, node_column_name='peer', node_index_column_name='peer_index')
	
	df = df.convert_dtypes()
	df.sort_values(['node_index', 'peer_index'], inplace=True)
	
	return df


def init_peers_nx_graph(df: pd.DataFrame) -> nx.Graph:
	g = nx.DiGraph()
	g.add_nodes_from(df['node'])
	g.add_edges_from(list(zip(df['node'], [p for p in df['peer'] if not pd.isna(p)])))
	
	if nx.is_isomorphic(g.to_undirected(reciprocal=True, as_view=True), g.to_undirected(reciprocal=False, as_view=True), node_match=lambda n1, n2: n1 is n2):
		g = g.to_undirected(reciprocal=True, as_view=True)
	
	return g


def sim_lats(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.sim_lats(raw_)
	
	df = pd.DataFrame(raw_)
	add_node_from_file(df)
	add_node_index_from_node(df)
	add_node_index_from_node(df, node_column_name='peer', node_index_column_name='peer_index')
	
	df = df.convert_dtypes()
	df.sort_values(['node_index', 'peer_index'], inplace=True)
	
	return df


def sim_lats_nx_graph(df: pd.DataFrame) -> nx.Graph:
	g = nx.DiGraph()
	g.add_nodes_from(pd.concat([df['node'], df['peer']]).unique())
	
	g.add_edges_from(
		df.apply(
			lambda r: (r['node'], r['peer'], {'latency_ms': r['latency_ms']} if 'latency_ms' in r else {}),
			axis='columns'
		)
	)
	
	return g


def data_dist(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.data_dist(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df)
	add_node_index_from_node(df)
	df = df.convert_dtypes()
	
	df['counts'] = df['counts'].apply(lambda x: tuple(x.items()))
	df = df.explode('counts')
	
	df['target'] = df['counts'].apply(lambda x: x[0])
	df['count'] = df['counts'].apply(lambda x: x[1])
	df.drop(columns='counts', inplace=True)
	
	df = df.convert_dtypes()
	df.sort_values(['node_index', 'target'], inplace=True)
	
	return df


def accs(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.accs(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df)
	add_node_index_from_node(df)
	df = df.convert_dtypes()
	df.sort_values(['round', 'node_index'], inplace=True)
	
	return df


def accs_mean(df: pd.DataFrame) -> pd.DataFrame:
	assert 'type' not in df.columns
	
	df_mean = df.groupby('round', as_index=False).agg({'accuracy': 'mean', 'mean_loss': 'mean'})
	df_mean['node'] = 'Mean'
	
	return df_mean


def sims(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.sims(raw_)
	
	df = pd.DataFrame(raw_)
	df.drop(columns=['file'], inplace=True)
	
	df = df.groupby('round')['sims'].agg(lambda ds: reduce(lambda da, db: da | db, ds)).reset_index()
	
	# TODO: assert all nodes' pairs are present in each round's similarities data.
	
	df['sims'] = df['sims'].apply(lambda d: tuple(d.items()))
	df = df.explode('sims')
	
	df['peers_pair'] = df['sims'].apply(lambda it: it[0])
	df['sim'] = df['sims'].apply(lambda it: it[1])
	df.drop(columns='sims', inplace=True)
	
	# Break `peers_pair` into `peer` and `target_peer`; also, augment with the reverse.
	
	df['peers_pair_t'] = df['peers_pair'].apply(lambda fs: tuple(fs))
	df['peer'] = df['peers_pair_t'].apply(lambda pp: pp[0])
	df['target_peer'] = df['peers_pair_t'].apply(lambda pp: pp[1])
	df.drop(columns=['peers_pair_t', 'peers_pair'], inplace=True)
	rev_sims_df = df.rename(columns={'peer': 'target_peer', 'target_peer': 'peer'})
	df = pd.concat([df, rev_sims_df], ignore_index=True)
	
	# Add the main diagonal, i.e. similarities of the same peers.
	
	peers = pd.unique(df[['peer', 'target_peer']].values.ravel())
	self_sims_df = pd.DataFrame(columns=df.columns)
	for round_ in df['round'].unique():
		r_self_sims_df = pd.DataFrame({'peer': peers, 'target_peer': peers, 'sim': 1, 'norm_sim': 1, 'round': round_})
		self_sims_df = pd.concat([self_sims_df, r_self_sims_df], ignore_index=True)
	df = pd.concat([df, self_sims_df], ignore_index=True)
	
	add_node_index_from_node(df, node_column_name='peer', node_index_column_name='peer_index')
	add_node_index_from_node(df, node_column_name='target_peer', node_index_column_name='target_peer_index')
	
	df = df.convert_dtypes()
	df.sort_values(['round', 'peer_index', 'target_peer_index'], inplace=True)
	df.reset_index(drop=True, inplace=True)
	
	df['peer'] = pd.Categorical(
		df['peer'],
		categories=df[['peer', 'peer_index']].sort_values(['peer_index'])['peer'].unique(),
		ordered=True
	)
	df['target_peer'] = pd.Categorical(
		df['target_peer'],
		categories=df[['target_peer', 'target_peer_index']].sort_values(['target_peer_index'])['target_peer'].unique(),
		ordered=True
	)
	
	return df


def son_sims(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.son_sims(raw_)
	
	df = pd.DataFrame(raw_)
	df.drop(columns=['file'], inplace=True)
	
	df = df.groupby('round')['sims'].agg(lambda ds: reduce(lambda da, db: da | db, ds)).reset_index()
	
	# TODO: assert all nodes' pairs are present in each round's similarities data.
	
	df['sims'] = df['sims'].apply(lambda d: tuple(d.items()))
	df = df.explode('sims')
	
	df['nodes_pair'] = df['sims'].apply(lambda it: it[0])
	df['sim'] = df['sims'].apply(lambda it: it[1])
	df.drop(columns='sims', inplace=True)
	
	df['node'] = df['nodes_pair'].apply(lambda pp: pp[0])
	df['son_node'] = df['nodes_pair'].apply(lambda pp: pp[1])
	df.drop(columns='nodes_pair', inplace=True)
	
	add_node_index_from_node(df, node_column_name='node', node_index_column_name='node_index')
	add_node_index_from_node(df, node_column_name='son_node', node_index_column_name='son_node_index')
	
	df = df.convert_dtypes()
	df.sort_values(['round', 'node_index', 'son_node_index'], inplace=True)
	df.reset_index(drop=True, inplace=True)
	
	df['node'] = pd.Categorical(
		df['node'],
		categories=df[['node', 'node_index']].sort_values(['node_index'])['node'].unique(),
		ordered=True
	)
	df['son_node'] = pd.Categorical(
		df['son_node'],
		categories=df[['son_node', 'son_node_index']].sort_values(['son_node_index'])['son_node'].unique(),
		ordered=True
	)
	
	return df


def agg_peers(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.agg_peers(raw_)
	
	df = pd.DataFrame(raw_)
	
	if 'node' not in df.columns:
		add_node_from_file(df)
	add_node_index_from_node(df)
	
	df = df.convert_dtypes()
	
	df = df.explode('peers')
	df.rename(columns={'peers': 'peer'}, inplace=True)
	df.dropna(subset=['peer'], inplace=True)
	add_node_index_from_node(df, node_column_name='peer', node_index_column_name='peer_index')
	
	df.sort_values(['round', 'node_index', 'peer_index'], inplace=True)
	
	df['node'] = pd.Categorical(
		df['node'],
		categories=df[['node', 'node_index']].sort_values(['node_index'])['node'].unique(),
		ordered=True
	)
	df['peer'] = pd.Categorical(
		df['peer'],
		categories=df[['peer', 'peer_index']].sort_values(['peer_index'])['peer'].unique(),
		ordered=True
	)
	
	return df


def zones(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.zones(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df, node_column_name='initiator_node')
	add_node_index_from_node(df, node_column_name='initiator_node', node_index_column_name='initiator_node_index')
	
	df.sort_values(['round', 'level', 'initiator_node_index'], inplace=True)
	df = df.convert_dtypes()
	
	df = df.explode('peers')
	df.rename(columns={'peers': 'peer'}, inplace=True)
	
	for (round_, level), group_df in df.groupby(['round', 'level']):
		for rl_representative in group_df['initiator_node']:
			df.loc[(df['round'] == round_)
			       & (df['level'] == level + 1)
			       & (df['peer'] == rl_representative),
			'peer_repr'] = f"{rl_representative} ({level + 1})"
			
			df.loc[(df['round'] == round_) & (df['level'] == level)
			       & (df['initiator_node'] == rl_representative),
			'initiator_node_repr'] = f"{rl_representative} ({level + 1})"
	
	df['peer_repr'] = df['peer_repr'].fillna(df['peer'])
	df['initiator_node_repr'] = df['initiator_node_repr'].fillna(df['initiator_node'])
	
	return df


def clusters(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.clusters(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df, node_column_name='representative_node')
	add_node_index_from_node(df, node_column_name='representative_node',
	                         node_index_column_name='representative_node_index')
	
	df = df.convert_dtypes()
	df.sort_values(['round', 'level', 'representative_node_index'], inplace=True)
	
	df = df.explode('peers')
	df.rename(columns={'peers': 'peer'}, inplace=True)
	
	for (round_, level), group_df in df.groupby(['round', 'level']):
		for rl_representative in group_df['representative_node']:
			df.loc[(df['round'] == round_)
			       & (df['level'] == level + 1)
			       & (df['peer'] == rl_representative),
			'peer_repr'] = f"{rl_representative} ({level + 1})"
			
			df.loc[(df['round'] == round_) & (df['level'] == level)
			       & (df['representative_node'] == rl_representative),
			'representative_node_repr'] = f"{rl_representative} ({level + 1})"
	
	df['peer_repr'] = df['peer_repr'].fillna(df['peer'])
	df['representative_node_repr'] = df['representative_node_repr'].fillna(df['representative_node'])
	
	return df


def super_clusters_representatives(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.super_clusters_representatives(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df, node_column_name='representative_node')
	add_node_index_from_node(df, node_column_name='representative_node', node_index_column_name='representative_node_index')
	
	df['super_clusters_representatives'] = df.apply(lambda r: r['super_clusters_representatives'] | {r['representative_node']}, axis='columns')
	
	return df


def searches(raw_: list[dict[str, Any]] | Path, evals: bool = True, sims_df: pd.DataFrame | None = None) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.searches(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df)
	add_node_index_from_node(df)
	df = df.convert_dtypes()
	df.sort_values(['node_index', 'round'], inplace=True)
	
	if 'target_k' in df.columns:
		df['k_diff'] = df.apply(lambda r: (len(r['result']) - (1 if r['node'] in r['result'] else 0)) - r['target_k'], axis='columns')
	
	if evals:
		assert sims_df is not None
		
		skips = set()
		searches_all_peers = set(df['node'].unique()) | set().union(*(d.keys() for d in df['result']))
		
		for idx, search_row in df.iterrows():
			if (search_row['node'], search_row['round']) in skips:
				continue
			
			rn_son_sims_df = sims_df[(sims_df['round'] == search_row['round']) & (sims_df['node' if 'node' in sims_df.columns else 'peer'] == search_row['node'])]
			base = rn_son_sims_df.set_index('son_node' if 'son_node' in rn_son_sims_df.columns else 'target_peer')['sim'].to_dict()
			
			assert all(peer in base.keys() for peer in searches_all_peers)
			
			relevant_entries = {target: sim for target, sim in base.items() if sim >= search_row['min_similarity']}
			relevant_entries = list(sorted(relevant_entries.items(), key=lambda t: t[1], reverse=True))
			if search_row['k'] > 0:
				relevant_entries = relevant_entries[:search_row['k']]
			relevant_entries = OrderedDict(relevant_entries)
			
			irrelevant_entries = {node: sim for node, sim in base.items() if node not in relevant_entries}
			retrieved_entries = search_row['result']
			
			irrelevant_entries = OrderedDict(sorted(irrelevant_entries.items(), key=lambda t: t[1], reverse=True))
			retrieved_entries = OrderedDict(sorted(retrieved_entries.items(), key=lambda t: t[1], reverse=True))
			
			relevant_nodes = set(relevant_entries.keys())
			irrelevant_nodes = set(irrelevant_entries.keys())
			retrieved_nodes = set(retrieved_entries.keys())
			
			tp = len(retrieved_nodes & relevant_nodes)
			fp = len(retrieved_nodes - relevant_nodes)
			fn = len(relevant_nodes - retrieved_nodes)
			tn = len(irrelevant_nodes) - fp
			
			recall = (tp / len(relevant_nodes)) if len(relevant_nodes) > 0 else math.nan
			precision = (tp / (tp + fp)) if (tp + fp) > 0 else math.nan
			accuracy = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else math.nan
			
			# NDCG
			
			if len(relevant_nodes) > 1:
				y_true = relevant_entries
				y_pred = retrieved_entries
				
				labels_order = list(searches_all_peers)
				y_true_relevance = [y_true.get(label, 0) for label in labels_order]
				y_pred_relevance = [y_pred.get(label, 0) for label in labels_order]
				
				ndcg = ndcg_score([y_true_relevance], [y_pred_relevance])
				if ndcg >= 1:
					assert ndcg - 1 <= 1e-7
					ndcg = 1
			else:
				ndcg = math.nan
			
			df.at[idx, 'recall'] = recall
			df.at[idx, 'precision'] = precision
			df.at[idx, 'accuracy'] = accuracy
			df.at[idx, 'ndcg'] = ndcg
	
	return df


def durs(raw_: list[dict[str, Any]] | Path) -> pd.DataFrame:
	if isinstance(raw_, Path):
		raw_ = raw.durs(raw_)
	
	df = pd.DataFrame(raw_)
	
	add_node_from_file(df)
	add_node_index_from_node(df)
	
	df['duration_seconds'] = df['duration'].apply(lambda dur: dur.total_seconds())
	
	df = df.convert_dtypes()
	df.sort_values(['node_index', 'round'], inplace=True)
	
	return df
