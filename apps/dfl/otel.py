from random import Random
from typing import Callable, Union, override, Awaitable

import grpc
import grpc.aio
from grpc.aio import UnaryUnaryClientInterceptor, UnaryStreamClientInterceptor, StreamUnaryClientInterceptor, StreamStreamClientInterceptor, ClientCallDetails, ServerInterceptor, ServicerContext
from grpc.aio._call import StreamStreamCall, StreamUnaryCall, UnaryStreamCall, UnaryUnaryCall
from grpc.aio._typing import RequestType, RequestIterableType, ResponseIterableType, ResponseType
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorServer, GrpcAioInstrumentorClient
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import IdGenerator

import conf


# See https://github.com/open-telemetry/opentelemetry-python/pull/4571; TODO.
class _RandomIdGenerator(IdGenerator):
	def __init__(self) -> None:
		self._rng = Random()
	
	def generate_span_id(self) -> int:
		span_id = self._rng.getrandbits(64)
		while span_id == trace.INVALID_SPAN_ID:
			span_id = self._rng.getrandbits(64)
		return span_id
	
	def generate_trace_id(self) -> int:
		trace_id = self._rng.getrandbits(128)
		while trace_id == trace.INVALID_TRACE_ID:
			trace_id = self._rng.getrandbits(128)
		return trace_id


class _ClientInterceptor(UnaryUnaryClientInterceptor, UnaryStreamClientInterceptor, StreamUnaryClientInterceptor, StreamStreamClientInterceptor):
	@override
	async def intercept_unary_unary(
			self,
			continuation: Callable[[ClientCallDetails, RequestType], UnaryUnaryCall],
			client_call_details: ClientCallDetails,
			request: RequestType
	) -> Union[UnaryUnaryCall, ResponseType]:
		span = trace.get_current_span()
		if span.is_recording():
			if client_call_details.metadata is not None:
				for k, v in client_call_details.metadata:
					span.set_attribute(f'rpc.grpc.request.metadata.{k}', v)
			
			span.set_attribute('rpc.client.request.size', request.ByteSize())
		
		res_call = await continuation(client_call_details, request)
		
		if span.is_recording():
			init_md = await res_call.initial_metadata()
			for k, v in init_md:
				span.set_attribute(f'rpc.grpc.response.metadata.initial.{k}', v)
			
			res = await res_call
			span.set_attribute('rpc.client.response.size', res.ByteSize())
			
			trail_md = await res_call.trailing_metadata()
			for k, v in trail_md:
				span.set_attribute(f'rpc.grpc.response.metadata.trailing.{k}', v)
		
		return res_call
	
	@override
	async def intercept_unary_stream(self, continuation: Callable[
		[ClientCallDetails, RequestType], UnaryStreamCall
	], client_call_details: ClientCallDetails, request: RequestType) -> Union[ResponseIterableType, UnaryStreamCall]:
		raise NotImplementedError  # TODO. Watch out for https://github.com/grpc/grpc/issues/31442.
	
	@override
	async def intercept_stream_unary(self, continuation: Callable[
		[ClientCallDetails, RequestType], StreamUnaryCall
	], client_call_details: ClientCallDetails, request_iterator: RequestIterableType) -> StreamUnaryCall:
		raise NotImplementedError  # TODO. Watch out for https://github.com/grpc/grpc/issues/31442.
	
	@override
	async def intercept_stream_stream(self, continuation: Callable[
		[ClientCallDetails, RequestType], StreamStreamCall
	], client_call_details: ClientCallDetails, request_iterator: RequestIterableType) -> Union[ResponseIterableType, StreamStreamCall]:
		raise NotImplementedError  # TODO. Watch out for https://github.com/grpc/grpc/issues/31442.


class _ServerInterceptor(ServerInterceptor):
	@override
	async def intercept_service(
			self,
			continuation: Callable[[grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]],
			handler_call_details: grpc.HandlerCallDetails
	) -> grpc.RpcMethodHandler:
		next_handler = await continuation(handler_call_details)
		
		# noinspection PyUnresolvedReferences
		if next_handler.unary_unary is not None:
			# noinspection PyUnresolvedReferences
			original_handler = next_handler.unary_unary
			
			async def wrapped_handler(request, context: ServicerContext):
				span = trace.get_current_span()
				if span.is_recording():
					for k, v in context.invocation_metadata():
						span.set_attribute(f'rpc.grpc.request.metadata.{k}', v)
					
					span.set_attribute('rpc.server.request.size', request.ByteSize())
				
				response = await original_handler(request, context)
				
				if span.is_recording():
					# TODO: record response metadata (initial and trailing).
					
					span.set_attribute('rpc.server.response.size', response.ByteSize())
				
				return response
			
			# noinspection PyUnresolvedReferences
			return grpc.unary_unary_rpc_method_handler(wrapped_handler, request_deserializer=next_handler.request_deserializer, response_serializer=next_handler.response_serializer)
		else:
			raise NotImplementedError  # TODO


def _instrument_client_intercepts() -> None:
	original_insecure = grpc.aio.insecure_channel
	original_secure = grpc.aio.secure_channel
	
	def insecure(*args, **kwargs):
		kwargs["interceptors"] = [_ClientInterceptor()] + kwargs.get("interceptors", [])
		return original_insecure(*args, **kwargs)
	
	def secure(*args, **kwargs):
		kwargs["interceptors"] = [_ClientInterceptor()] + kwargs.get("interceptors", [])
		return original_secure(*args, **kwargs)
	
	grpc.aio.insecure_channel = insecure
	grpc.aio.secure_channel = secure


def _instrument_server_intercepts() -> None:
	original_server = grpc.aio.server
	
	def server(*args, **kwargs):
		kwargs['interceptors'] = [_ServerInterceptor()] + kwargs.get("interceptors", [])
		return original_server(*args, **kwargs)
	
	grpc.aio.server = server


def instrument() -> None:
	tracer_provider = TracerProvider(resource=Resource({'service.name': conf.NAME}), id_generator=_RandomIdGenerator())
	tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint='otel-collector:4317', insecure=True)))
	trace.set_tracer_provider(tracer_provider)
	
	GrpcAioInstrumentorClient().instrument()
	GrpcAioInstrumentorServer().instrument()
	
	_instrument_client_intercepts()
	_instrument_server_intercepts()
