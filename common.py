import ctypes

__all__ = [
    "EngineRuntime",
]

_lib = ctypes.CDLL("/home/matthew/inference-engine-2/zig-out/lib/libinference.so")

_lib.enqueue_high_priority_embedding_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.enqueue_high_priority_embedding_request.restype = ctypes.c_int32

_lib.enqueue_low_priority_embedding_request.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.enqueue_low_priority_embedding_request.restype = ctypes.c_int32

_lib.load_shared_memory_from_fd.argtypes = [ctypes.c_int32, ctypes.c_size_t]
_lib.load_shared_memory_from_fd.restype = ctypes.c_void_p

_lib.unload_shared_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.unload_shared_memory.restype = None

class EngineRuntime:
    def __init__(self, fd: int, size: int):
        self.fd = fd
        self.size = size

    def __enter__(self):
        self.addr = _lib.load_shared_memory_from_fd(self.fd, self.size)
        return self
    
    def __exit__(self, _exc_type, _exc_value, _traceback):
        _lib.unload_shared_memory(self.addr, self.size)
        return False

    def enqueue_high_priority_embedding_request(self, text: str) -> int:
        c_text = ctypes.create_string_buffer(text.encode())
        return int(_lib.enqueue_high_priority_embedding_request(self.addr, c_text.value))

    def enqueue_low_priority_embedding_request(self, text: str) -> int:
        c_text = ctypes.create_string_buffer(text.encode())
        return int(_lib.enqueue_low_priority_embedding_request(self.addr, c_text.value))

