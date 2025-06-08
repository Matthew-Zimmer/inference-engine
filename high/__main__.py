import sys, asyncio
from .common import EngineRuntime

fd = int(sys.argv[1])
size = int(sys.argv[2])

async def main():
    with EngineRuntime(fd, size) as rt:
        text = "embedding request from high script" 
        pages = [len(text)]
        print(f"[HIGH SCRIPT] - Sending request for \"{text}\" {pages}")
        res = await rt.enqueue_high_priority_chunked_embedding_request(text, pages)
        print(f"[HIGH SCRIPT] - Got response")
        print(f"[HIGH SCRIPT] - {res}")

asyncio.run(main())

