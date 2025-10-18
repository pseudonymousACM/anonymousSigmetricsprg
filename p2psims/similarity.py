from scipy.spatial.distance import cosine as scipy_cosine_distance, euclidean as scipy_euclidean_distance


async def cosine(u, v) -> float:
	cosine_v = 1 - scipy_cosine_distance(u, v)
	return (cosine_v + 1) / 2  # Min-max normalization from [-1, 1] to [0, 1].


async def euclidean(u, v) -> float:
	return 1 / (1 + scipy_euclidean_distance(u, v))
