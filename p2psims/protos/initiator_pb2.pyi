from google.protobuf import empty_pb2 as _empty_pb2
from p2psims.protos import peer_pb2 as _peer_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class JoinZoneDecision(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACCEPT: _ClassVar[JoinZoneDecision]
    REJECT: _ClassVar[JoinZoneDecision]
    INFORM: _ClassVar[JoinZoneDecision]
ACCEPT: JoinZoneDecision
REJECT: JoinZoneDecision
INFORM: JoinZoneDecision

class JoinZoneRequest(_message.Message):
    __slots__ = ("decision", "initiator_address", "data")
    DECISION_FIELD_NUMBER: _ClassVar[int]
    INITIATOR_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    decision: JoinZoneDecision
    initiator_address: str
    data: _peer_pb2.Data
    def __init__(self, decision: _Optional[_Union[JoinZoneDecision, str]] = ..., initiator_address: _Optional[str] = ..., data: _Optional[_Union[_peer_pb2.Data, _Mapping]] = ...) -> None: ...
