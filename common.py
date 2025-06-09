import ctypes, asyncio, os
from dataclasses import dataclass

__all__ = [
    "EngineRuntime",
    "ChunkedEmbeddingResult",
]

class EventFdsInfo(ctypes.Structure):
    _fields_ = [("len", ctypes.c_size_t), ("data", ctypes.POINTER(ctypes.c_int32))]

    def to_python(self) -> list[int]:
        return [int(self.data[i]) for i in range(int(self.len))] 

@dataclass
class ChunkedEmbeddingResult:
    large_chunk_embeddings: list[list[float]]
    small_chunk_embeddings: list[list[float]]
    page_chunk_embeddings: list[list[float]]

DIM = 768
dummy_embedding = [0.0] * DIM
LIB_ERROR = 0xffffffffffffffff

class _ChunkedEmbeddingResult(ctypes.Structure):
    _fields_ = [
        ("large_chunk_embeddings_len", ctypes.c_size_t),
        ("large_chunk_embeddings_data", ctypes.POINTER(ctypes.c_float)),
        ("small_chunk_embeddings_len", ctypes.c_size_t),
        ("small_chunk_embeddings_data", ctypes.POINTER(ctypes.c_float)),
        ("page_chunk_embeddings_len", ctypes.c_size_t),
        ("page_chunk_embeddings_data", ctypes.POINTER(ctypes.c_float)),
    ]

    # TODO: how perfomant is this?
    def to_python(self) -> ChunkedEmbeddingResult:
        large_chunks_len = int(self.large_chunk_embeddings_len) 
        large_chunks = [dummy_embedding] * large_chunks_len
        for i in range(large_chunks_len):
            for j in range(DIM):
                large_chunks[i][j] = float(self.large_chunk_embeddings_data[i * DIM + j])

        small_chunks_len = int(self.small_chunk_embeddings_len)
        small_chunks = [dummy_embedding] * small_chunks_len
        for i in range(small_chunks_len):
            for j in range(DIM):
                small_chunks[i][j] = float(self.small_chunk_embeddings_data[i * DIM + j])

        page_chunks_len = int(self.page_chunk_embeddings_len)
        page_chunks = [dummy_embedding] * page_chunks_len
        for i in range(page_chunks_len):
            for j in range(DIM):
                page_chunks[i][j] = float(self.page_chunk_embeddings_data[i * DIM + j])

        return ChunkedEmbeddingResult(large_chunks, small_chunks, page_chunks)
        
_lib = ctypes.CDLL("/home/matthew/inference-engine-2/zig-out/lib/libinference-engine-runtime.so")

_lib.load_shared_memory_from_fd.argtypes = [ctypes.c_int32, ctypes.c_size_t]
_lib.load_shared_memory_from_fd.restype = ctypes.c_void_p

_lib.unload_shared_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.unload_shared_memory.restype = None

_lib.event_fds.argtypes = [ctypes.c_void_p]
_lib.event_fds.restype = EventFdsInfo

_lib.enqueue_high_priority_chunked_embedding_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
_lib.enqueue_high_priority_chunked_embedding_request.restype = ctypes.c_size_t

_lib.enqueue_low_priority_chunked_embedding_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
_lib.enqueue_low_priority_chunked_embedding_request.restype = ctypes.c_size_t

_lib.chunked_request_event_fd.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.chunked_request_event_fd.restype = ctypes.c_int32

_lib.chunked_request_result.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.chunked_request_result.restype = _ChunkedEmbeddingResult

_lib.chunked_request_deinit.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.chunked_request_deinit.restype = None

async def _wait_for_fd_signal(fd: int):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def on_ready():
        if not future.done():
            future.set_result(None)
        loop.remove_reader(fd)
    
    loop.add_reader(fd, on_ready)
    return await future

class EngineRuntime:
    def __init__(self, fd: int, size: int):
        self.fd = fd
        self.size = size

    def __enter__(self):
        self.addr = _lib.load_shared_memory_from_fd(self.fd, self.size)
        self.event_fds = _lib.event_fds(self.addr).to_python()
        return self
    
    def __exit__(self, _exc_type, _exc_value, _traceback):
        _lib.unload_shared_memory(self.addr, self.size)
        loop = asyncio.get_running_loop()
        for efd in self.event_fds:
            try:
                loop.remove_reader(efd)
            except:
                pass
            os.close(efd)
        os.close(self.fd)
        return False

    async def enqueue_high_priority_chunked_embedding_request(self, text: str, pages: list[int]) -> ChunkedEmbeddingResult:
        c_text = ctypes.create_string_buffer(text.encode())
        pages_array_type = ctypes.c_uint64 * len(pages)
        pages_array = pages_array_type(*pages)
        
        request = _lib.enqueue_high_priority_chunked_embedding_request(self.addr, c_text.value, ctypes.c_size_t(len(pages)), pages_array)
        if int(request) == LIB_ERROR: raise RuntimeError()

        event_fd = int(_lib.chunked_request_event_fd(self.addr, request))
        await _wait_for_fd_signal(event_fd)

        res = _lib.chunked_request_result(self.addr, request).to_python()
        _lib.chunked_request_deinit(self.addr, request)

        return res

    async def enqueue_low_priority_chunked_embedding_request(self, text: str, pages: list[int]) -> ChunkedEmbeddingResult:
        c_text = ctypes.create_string_buffer(text.encode())
        pages_array_type = ctypes.c_uint64 * len(pages)
        pages_array = pages_array_type(*pages)
        
        request = _lib.enqueue_low_priority_chunked_embedding_request(self.addr, c_text.value, ctypes.c_size_t(len(pages)), pages_array)
        if int(request) == LIB_ERROR: raise RuntimeError()

        event_fd = int(_lib.chunked_request_event_fd(self.addr, request))
        await _wait_for_fd_signal(event_fd)

        res = _lib.chunked_request_result(self.addr, request).to_python()
        _lib.chunked_request_deinit(self.addr, request)

        return res

