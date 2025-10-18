from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("datum", "k", "min_similarity", "sequential_top_k")
    DATUM_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    MIN_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    SEQUENTIAL_TOP_K_FIELD_NUMBER: _ClassVar[int]
    datum: bytes
    k: int
    min_similarity: float
    sequential_top_k: bool
    def __init__(self, datum: _Optional[bytes] = ..., k: _Optional[int] = ..., min_similarity: _Optional[float] = ..., sequential_top_k: bool = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("scores",)
    class ScoresEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    SCORES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.ScalarMap[str, float]
    def __init__(self, scores: _Optional[_Mapping[str, float]] = ...) -> None: ...

class NodeIdentity(_message.Message):
    __slots__ = ("address", "level")
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    address: str
    level: int
    def __init__(self, address: _Optional[str] = ..., level: _Optional[int] = ...) -> None: ...

class PromoteToSuperClusterRepresentativeRequest(_message.Message):
    __slots__ = ("super_clusters_representatives_addresses",)
    SUPER_CLUSTERS_REPRESENTATIVES_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    super_clusters_representatives_addresses: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, super_clusters_representatives_addresses: _Optional[_Iterable[str]] = ...) -> None: ...

class ImmediateSearchRequest(_message.Message):
    __slots__ = ("datum",)
    DATUM_FIELD_NUMBER: _ClassVar[int]
    datum: bytes
    def __init__(self, datum: _Optional[bytes] = ...) -> None: ...

class ImmediateSearchResponse(_message.Message):
    __slots__ = ("score",)
    SCORE_FIELD_NUMBER: _ClassVar[int]
    score: float
    def __init__(self, score: _Optional[float] = ...) -> None: ...
