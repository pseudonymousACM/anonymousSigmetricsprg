import torch


def topk_mag(t: torch.Tensor, k: int, force_sparse: bool = False) -> torch.Tensor:
	numel = t.numel()
	assert k <= numel
	if k == numel:
		if force_sparse and not t.is_sparse:
			t = t.to_sparse()
		return t
	
	t_flat = t.flatten()
	
	# This `.abs()` call triggers a new tensor memory allocation (i.e. a deep copy).
	# One solution to avoid this deep copy is to take the topk once with `largest=True` and once with `largest=False`, then combine their results, using a merge sort approach or another topk on the combined elements;
	# but, should watch out for the duplicate indices which can happen if k > numel/2, or there are duplicate values in the tensor; this watch-out would introduce a Python loop which would kill the performance when k is large.
	t_flat_abs = t_flat.abs()
	
	_, indices = t_flat_abs.topk(k=k, largest=True, sorted=False)
	vals = t_flat[indices]
	
	indices = torch.stack(torch.unravel_index(indices, t.size()))
	t_topk_sparse = torch.sparse_coo_tensor(indices, vals, t.size())
	
	t_topk_sparse._coalesced_(True)
	
	return t_topk_sparse


def topk_mag_tensors(ts: list[torch.Tensor], k: int) -> list[torch.Tensor]:
	numel = sum(t.numel() for t in ts)
	assert k <= numel
	if k == numel:
		return ts
	
	# TODO: values and indices can be handled together if PyTorch's topk supported sparse tensors (CUDA mainly).
	
	t0_flat = ts[0].flatten()
	topk = topk_mag(t0_flat, k=min(k, t0_flat.numel()), force_sparse=True)
	topk_vals, topk_indices = topk.values(), topk.indices().squeeze(dim=0)
	topk_ts_indices = torch.full_like(topk_indices, 0)
	for i, t in enumerate(ts[1:], start=1):
		t_flat = t.flatten()
		t_topk = topk_mag(t_flat, k=min(k, t_flat.numel()), force_sparse=True)
		t_topk_vals, t_topk_indices = t_topk.values(), t_topk.indices().squeeze(dim=0)
		t_topk_ts_indices = torch.full_like(t_topk_indices, i)
		
		comb_vals = torch.cat((topk_vals, t_topk_vals))
		comb_indices = torch.cat((topk_indices, t_topk_indices))
		comb_ts_indices = torch.cat((topk_ts_indices, t_topk_ts_indices))
		
		comb_topk = topk_mag(comb_vals, k=min(k, comb_vals.numel()), force_sparse=True)
		comb_topk_vals, comb_topk_indices = comb_topk.values(), comb_topk.indices().squeeze(dim=0)
		
		topk_vals = comb_topk_vals
		topk_indices = comb_indices[comb_topk_indices]
		topk_ts_indices = comb_ts_indices[comb_topk_indices]
	
	rs = []
	for i in range(len(ts)):
		mask = topk_ts_indices == i
		rs.append((topk_vals[mask], topk_indices[mask]))
	
	for i, (vals, indices) in enumerate(rs):
		r_indices = torch.stack(torch.unravel_index(indices, ts[i].size()))
		r = torch.sparse_coo_tensor(r_indices, vals, ts[i].size())
		r._coalesced_(True)
		rs[i] = r
	
	return rs


def topp_mag(t: torch.Tensor, p: float, force_sparse: bool = False) -> torch.Tensor:
	assert 0 <= p <= 1
	
	numel = t.numel()
	k = round(numel * p)
	
	return topk_mag(t, k=k, force_sparse=force_sparse)


def topp_mag_tensors(ts: list[torch.Tensor], p: float) -> list[torch.Tensor]:
	assert 0 <= p <= 1
	
	numel = sum(t.numel() for t in ts)
	k = round(numel * p)
	
	return topk_mag_tensors(ts, k=k)


def _is_sparse_more_mem_eff_formula(nnz: int, elem_size: int, numel: int, ndim: int) -> bool:
	dense_size = numel * elem_size
	sparse_size = nnz * (elem_size + ndim * 8)  # 8 bytes per index.
	
	return sparse_size < dense_size


def is_sparse_more_mem_eff(t: torch.Tensor) -> bool:
	if t.numel() == 0:
		return False
	
	if t.is_sparse:
		nnz = t.values().numel()
	else:
		nnz = torch.count_nonzero(t).item()
	
	numel = t.numel()
	elem_size = t.element_size()
	ndim = t.ndim
	
	return _is_sparse_more_mem_eff_formula(nnz, elem_size, numel, ndim)


def mean_tensors(ts: list[torch.Tensor]) -> torch.Tensor:
	return torch.mean(torch.stack(ts), dim=0)


def mean_sparse_tensors(ts: list[torch.Tensor]) -> torch.Tensor:
	res = ts[0].clone()
	for t in ts[1:]:
		res.add_(t)
	
	res.divide_(len(ts))
	return res


def mean_weighted_sparse_tensors(ts: list[tuple[torch.Tensor, float]]) -> tuple[torch.Tensor, float]:
	res = torch.zeros_like(ts[0][0])
	res_w = 0
	for (t, w) in ts:
		res.add_(t, alpha=w)
		res_w += w
	
	res.divide_(res_w)
	return res, res_w
