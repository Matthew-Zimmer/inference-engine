import sys
from .common import EngineRuntime

fd = int(sys.argv[1])
size = int(sys.argv[2])

with EngineRuntime(fd, size) as rt:
    text = "embedding request from low script" 
    pages = [len(text)]
    pass
    # print("[LOW SCRIPT] - request offset:", rt.enqueue_low_priority_chunked_embedding_request(text, pages))

