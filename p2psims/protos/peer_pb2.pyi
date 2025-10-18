from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SuggestJoinZoneRequest(_message.Message):
    __slots__ = ("initiator_address",)
    INITIATOR_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    initiator_address: str
    def __init__(self, initiator_address: _Optional[str] = ...) -> None: ...

class SuggestClusterRepresentativeRequest(_message.Message):
    __slots__ = ("force", "data", "peers_addresses")
    FORCE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    PEERS_ADDRESSES_FIELD_NUMBER: _ClassVar[int]
    force: bool
    data: bytes
    peers_addresses: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, force: bool = ..., data: _Optional[bytes] = ..., peers_addresses: _Optional[_Iterable[str]] = ...) -> None: ...

class SuggestClusterRepresentativeResponse(_message.Message):
    __slots__ = ("accept",)
    ACCEPT_FIELD_NUMBER: _ClassVar[int]
    accept: bool
    def __init__(self, accept: bool = ...) -> None: ...

class Data(_message.Message):
    __slots__ = ("peers_addresses_data",)
    class PeersAddressesDataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bytes
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
    PEERS_ADDRESSES_DATA_FIELD_NUMBER: _ClassVar[int]
    peers_addresses_data: _containers.ScalarMap[str, bytes]
    def __init__(self, peers_addresses_data: _Optional[_Mapping[str, bytes]] = ...) -> None: ...

class MigrateZoneInitiatorRequest(_message.Message):
    __slots__ = ("new_zone_initiator_address",)
    NEW_ZONE_INITIATOR_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    new_zone_initiator_address: str
    def __init__(self, new_zone_initiator_address: _Optional[str] = ...) -> None: ...

class PromoteToZoneInitiatorRequest(_message.Message):
    __slots__ = ("peers_addresses_data",)
    class PeersAddressesDataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Data
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Data, _Mapping]] = ...) -> None: ...
    PEERS_ADDRESSES_DATA_FIELD_NUMBER: _ClassVar[int]
    peers_addresses_data: _containers.MessageMap[str, Data]
    def __init__(self, peers_addresses_data: _Optional[_Mapping[str, Data]] = ...) -> None: ...
