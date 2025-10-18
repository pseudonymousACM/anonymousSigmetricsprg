#!/usr/bin/env sh

set -e
cd "$(dirname "$0")/../../"

python -m grpc_tools.protoc -I . --python_out . --grpc_python_out . --pyi_out . ./p2psims/protos/initiator.proto ./p2psims/protos/peer.proto ./p2psims/protos/representative.proto