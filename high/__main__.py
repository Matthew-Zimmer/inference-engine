import sys, asyncio, time
from .common import EngineRuntime

fd = int(sys.argv[1])
size = int(sys.argv[2])


async def main():
    with EngineRuntime(fd, size) as rt:
        text = "embedding request from high script" 
        pages = [len(text)]
        print(f'[HIGH SCRIPT] - Sending request for "{text}" {pages}')
        start = time.time()
        res = await rt.enqueue_high_priority_chunked_embedding_request(text, pages)
        end = time.time()
        print(f"[HIGH SCRIPT] - Got response in {(end - start) * 1e6:.02f}us")
        with open("high.log", "w") as f:
            f.write(f"[HIGH SCRIPT] - {res}")


asyncio.run(main())
