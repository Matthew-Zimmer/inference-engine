import sys
from .common import EngineRuntime

fd = int(sys.argv[1])
size = int(sys.argv[2])

with EngineRuntime(fd, size) as rt:
    text = "embedding request from high script" 
    pages = [len(text)]
    print("[HIGH SCRIPT] - request offset:", rt.enqueue_high_priority_chunked_embedding_request(text, pages))

