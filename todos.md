# Fast TODOs

> Slow, carefully documented todos would reside in the GitHub issues.

## Baseline Implementation

- Add links creation logic.
- Self-harassing search; no recursion.
- Multiple representatives per cluster.
- Top-level initiator replication logic.
- Support overlapping clusters.
- Other peer selection methods.
- Adaptive clustering.

## Improvements

- Use stub or channel interceptor for adding the `x-address`, `x-level`, and `x-session` invocation metadata.
- Use error logging interceptor on the clients and the server.
- Auto manage memory instances of gRPC stubs and channels.
	- Cleanup obsolete memory objects; e.g. zone initiator's data of its peers after assigning representatives.
	- Reuse gRPC channels and stubs of the underlying nodes (the client interceptors change per node currently; should overcome that).
- Optimize suggest join zone process by omitting sending redundant requests.
- Offload meta nodes routing logic to gRPC interceptors, if possible.
- Plant the `dict` version of `context.invocation_metadata()` into the `context` for reuse, by an interceptor.
- Remove extraneous debug logs. Add info logs reflecting the created semantic overlay network topology.
- Avoid casts to the np types for the similarity and aggregation methods.

## Features

- Peer discovery interface.
- Minimum similarity agnostic search functionality; only required to adhere to the given `k`.
- Add `noised_preferred_zone_size` zone partitioning method.

## Bugs

- Evaluate the sequential top-k search by locally simulating the algorithm.
- Do state management in the node; otherwise, out of sync nodes might call out-of-state operations; e.g. `JoinZone` call on an initiator that has already done clustering. 