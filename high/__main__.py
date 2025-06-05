import sys
from .common import EngineRuntime

fd = int(sys.argv[1])
size = int(sys.argv[2])

with EngineRuntime(fd, size) as rt:
    print("[HIGH SCRIPT] - request offset:", rt.enqueue_high_priority_embedding_request("embedding request from high script"))

