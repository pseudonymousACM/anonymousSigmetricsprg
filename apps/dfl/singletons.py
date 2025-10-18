import grpc
import p2psims.node
import torch
from dfl import Communicator
from dfl.communication.caching import StrongSubscribeCachingCommunicatorDecorator
from dfl.communication.grpc import GrpcCommunicator
from dfl.node.torch import Node as DflNode

from cdfl import CDflNode
from exts.dfl_exts import Server as CflServer, Client as CflClient

grpc_server: grpc.aio.Server | None = None

# FIXME: should make everywhere use this instead of `p2psims_node`, right?
p2psims_sessions_nodes = p2psims.node.SessionMultiplexingServicer()
p2psims_node: p2psims.Node | None = None

base_dfl_communicator: GrpcCommunicator | None = None
dfl_communicator: Communicator | None = None
dfl_caching_communicator: StrongSubscribeCachingCommunicatorDecorator | None = None

dfl_node: DflNode | None = None
cdfl_node: CDflNode | None = None

cfl_server: CflServer | None = None
cfl_client: CflClient | None = None

round_ = -1

round_train_accumulated_params_grads: list[torch.Tensor] | None = None
