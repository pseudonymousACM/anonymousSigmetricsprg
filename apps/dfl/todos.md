# TODOs

## Bugs

- Update `FAIR_SPREAD_ON_GPUS` documentation (in `README.md` and `.env.example`).
- Rename `FAIR_SPREAD_ON_GPUS` to `EVEN_SPREAD_CUDA_GPUS`.

- Definitely having a memory leak somewhere.
	- Test the training routine (many epochs).
- `SuggestJoinZone` taking too much time!
	- Test closing all gRPC channels at each round (practically with the GC option, before the actual GC).
- `Other threads are currently calling into gRPC, skipping fork() handlers` warning (gRPC and sync manager server start on `n0`).
- `WARNING: All log messages before absl::InitializeLog() is called are written to STDERR`
- Retry indefinitely for sending the traces' logs (it currently gives up at some point).
- Error raise in the clustering function is not propagated; i.e., it is silently ignored! (check again after adding graceful shutdown timeout to the gRPC servers and channels.)
- When fair spreading on GPUs using the device ID (`FAIR_SPREAD_ON_GPUS=id`), the VRAM usage skyrockets compared to spreading using `CUDA_VISIBLE_DEVICES`.
- `Error in sys.excepthook: Original exception was: <EOF>`
- CFL does not work with gradient-based search data, because the server does not receive the clients' gradients beside their models' parameters, nor receives the clients' search data themselves.
- Monitoring similarities doesn't work in CFL.
	- Seems working, but would raise device mismatch error if `SON_DATA_DEVICE` is not set (happens more promptly when having a `val_loader`).
- Mean + Stds. plots are displayed incorrectly if a number is NAN; see https://stackoverflow.com/questions/61494278/plotly-how-to-make-a-figure-with-multiple-lines-and-shaded-area-for-standard-de#comment137567930_69594497.
- The linear relation between `ASYNCIO_IO_THREADS` and VRAM usage.
	- Maybe a bug in the caching communicator when having lots of concurrency?
- LR notebook uses too much CPU and little GPU when setting CUDA device.
- Some configurations are applied in the `main` module which is not activated in the `lr` and `bs` and `prune` notebooks.

## Features

- Add documentations about configuring ARP table size and Linux process max keys, when spawning many containers.

- Add SON ideal top-k centralized aggregator method (same as ideal top-k, but compare with SON renewal rounds' data).
- Fight test over-fitting via:
	- Validation dataset & early stopping
		- https://octaipipe.medium.com/early-stopping-strategies-for-federated-learning-fl-62100c4c7b2b
		- https://openreview.net/forum?id=ptucbwZ1xA&referrer=%5Bthe%20profile%20of%20Yingwei%20Zhang%5D(%2Fprofile%3Fid%3D~Yingwei_Zhang1)
	- Regularization: L1, L2 (done ✅️), dropout, weight decay.
	- Data augmentation
- A measure for the SON hierarchical clusters/clustering quality.
- A simple greedy minimum similarity tuning approach using only the last search's results.
- Add affinity-propagation preference configuration.
- Recall at k in the isotonic method.
- Add the PENS method.

- Add DBSCAN clustering method.
- Add the clustering method of affinity-propagation with gossiping similarities' median for use as the affinity propagation's preference parameter.
- Add `ideal-data-dist` peers selection method: select based on data distribution similarity.
- Add K warm-up rounds (configurable); random gossip without any search.
- Add a ViT model.
- Clipped gradients (by norm.) search data; probably only affects Euclidean-based methods as Cosine already does the normalization.
- Add precision at k and recall at k search metrics.
- What value to choose as k? Evaluate the received model on our training dataset, and if it passed (its loss <= our own loss), then choose for aggregation. Meanwhile, interpolate k to tidy/expand the search space.
	- This idea (loss) alone is also a peers selection method, combined with the `initial-peers` method.
- Add an option to disable monitoring stuff.
	- DFL node calculating the waiting time per pull.
- A combination of sequential and parallel search to guarantee not over reaching into the network for just k results.
- Outlier detection on the search result; useful for after the sequential search, and for adaptive k selection.
- Add a LLM model (e.g. bert) with an applicable federated dataset.
- The isotonic method; what if not found k? Retry the search until a criteria is met (e.g. found at least one peer)?
- Add a third `FAIR_SPREAD_ON_GPUS` method, using Docker GPU device lend setting.
- Add shuffle topology configuration.
- Add an option to do direct topp search data construction which is apparently way faster but more memory hungry.
- Shapley value to find the optimal aggregation peers; find the relation between shapley values and similarities (visualize).
	- Add a shapley value peers selection method.

- Trace gRPC, e.g. by OpenTelemetry.
	- Add round as a span attribute.
	- Track search requests and count the number of contacted peers.
	- Recall per number of contacted peers plot.
	- Add an option to filter sampled methods.

- Add configurable DFL aggregation methods.
	- FedAvg
	- FedAvg weighted by similarities?
	- FedProx (adds a regularization term)
	- FedNova (normalizes based on local computations, e.g. epochs)
	- SCAFFOLD
	- FedDyn (adds a local regularization term)
	- FedBN (local batch normalization layers)
	- FedOpt

- Log the similarities after aggregation?
- Flooding/gossip search implementation?
- Hierarchical FL vs our work?
- Add Gaussian noise-based feature imbalance.
- Simulate latencies based on an Internet topology generator.
- Add the following topology generators for the initial peers:
	- See https://chatgpt.com/c/67f51ee3-d1a8-8007-a018-c9fb39a670df.
	- [GT-ITM](https://sites.cc.gatech.edu/projects/gtitm/)
	- Power-law topologies
	- SQUARE
	- near-RRG
- What to do with batch normalization layers in the ResNet18 model? Do not consider the BN layers for the DFL operations (is this idea the FedBN aggregation method?)?

## Improvements

- Ensure the calculations of the search evaluation metrics.
	- Validate the NDCG calculation.
- Remove `one-ten` data distribution in favor of `non-iid-1` (or `pretty-non-iid-k` where k equals the total number of unique labels)?
- `SKIP_DATA` feature, to skip the `data` service.
- Optimize the `data` service to not do data partitioning.
- Validate the simulated latencies by `ping`.
- Use the logger name instead of the `type` extra.
- Refactor data distribution methods into their own module and functions.
- Skip Otel collector; log traces directly on the nodes.
- The similarities log format can be simplified, if every node is not required to compute a similarity pair that does not include self.
- Remove the `eval` postfix for variables inside `conf_eval`.
	- Move configurations evaluations out of `conf_eval` into their respective modules.
- Rely solely on `singletons.dfl_node._round` after adding an option to manually increase it after calling `step_round`.
- Sync rounds based on the rounds age configuration; i.e. the maximum rounds difference between all pairs of nodes should be the rounds age.
- Smart `SEARCH_NO_PROBE_DURATION`; select based on a multiple of the maximum latency path in the initial peers graph.
- Omit the `otel-collector` service in case of `TRACE_RPCS` set to negative.
- Make the generated Compose file (more precisely, its inherently built image) dependent on the Git commit hash.
- Sync at the start of the rounds, not the ends; this would optimize by doing one less sync (the final sync).
- Move the preferred zone size configuration and the search no probe duration into its peers-selection configuration.
- Define and use a standard for including extra options with configuration values (e.g. `PEERS_SELECTION=top-k-preffered_zone_size=5,no_probe_duration=3.5`)
- Set the default values of the configurations to their most vanilla versions.
- Use `torch.nn.utils.parameters_to_vector` and `torch.nn.utils.vector_to_parameters`, instead of custom implementations.
- Maybe the dnsmasq service is unnecessary and all does DNS errors was due to ARP table overflow.
- Load and partition the datasets using the Flower library.
- Use PyTorch lighting?
- Separate train and evaluation batch sizes' configurations.
- Aggregations induce bias into the similarities; need to reduce the effect ensuring that the similarities are as true as possible, even after aggregating with an unideal peer.
- Reaching floating-point encoding limits in computing the similarities when models are initialized homogeneously (and/or after many rounds of constant aggregations, which is a less prominent cause).
- Can disable the pre-agg. test eval. in some methods, e.g. no-one (local) peers' selection method.
- Cleanup (remove) the `data_dist/` directory related to the `PREPATE_DATA_DISTRIBUTION` config. after the run.
- Rename `par-topk-ideal-th` peers' selection method to `par-topk-idealth`.