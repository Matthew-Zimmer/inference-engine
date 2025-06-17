import sys, asyncio, time, random
from typing import Any, Callable, Coroutine
import numpy as np
from .common import EngineRuntime

fd = int(sys.argv[1])
size = int(sys.argv[2])


words = [
    "wrist",
    "stunning",
    "art",
    "plead",
    "analyst",
    "punch",
    "fireplace",
    "kidnap",
    "valid",
    "mouth",
    "medal",
    "pour",
    "field",
    "warn",
    "meet",
    "priority",
    "proportion",
    "hill",
    "watch",
    "dressing",
    "border",
    "emphasis",
    "novel",
    "production",
    "rabbit",
    "traction",
    "garlic",
    "press",
    "mosque",
    "sin",
    "premature",
    "raw",
    "speaker",
    "dialect",
    "reign",
    "hair",
    "promotion",
    "castle",
    "knot",
    "jacket",
    "able",
    "agent",
    "rub",
    "equinox",
    "storage",
    "morsel",
    "red",
    "delivery",
    "episode",
    "fraction",
    "chapter",
    "AIDS",
    "anniversary",
    "confusion",
    "brave",
    "endure",
    "sword",
    "save",
    "advance",
    "rabbit",
    "tenant",
    "quotation",
    "fluctuation",
    "buffet",
    "stimulation",
    "identification",
    "resist",
    "top",
    "friend",
    "instruction",
    "conductor",
    "ideology",
    "mood",
    "hole",
    "talk",
    "day",
    "sandwich",
    "master",
    "spite",
    "deficit",
    "heir",
    "free",
    "have",
    "accompany",
    "confession",
    "contemporary",
    "aquarium",
    "embox",
    "multiply",
    "profession",
    "judgment",
    "judge",
    "compliance",
    "damage",
    "battlefield",
    "convince",
    "boat",
    "cat",
    "payment",
    "oak",
]

def random_text():
    text = list[str]()
    for _ in range(1000):
        text.append(random.choice(words))
    return " ".join(text)

async def main():
    with EngineRuntime(fd, size) as rt:
        text = sys.argv[3]
        pages = [len(text)]
        _ = await rt.enqueue_high_priority_chunked_embedding_request(text, pages)
        _ = await rt.enqueue_high_priority_chunked_embedding_request(text, pages)


async def pool(n: int, duration_seconds: int, f: Callable[[], Coroutine]):
    start_time = time.time()
    task_id_counter = 0
    in_flight: set[asyncio.Task[Any]] = { asyncio.create_task(f()) for _ in range(n) }

    while True:
        done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            _ = task.result()

        while len(in_flight) < n and time.time() - start_time < duration_seconds:
            task_id_counter += 1
            task = asyncio.create_task(f())
            in_flight.add(task)

        if time.time() - start_time >= duration_seconds:
            if not in_flight:
                break

    return task_id_counter

async def embedding_task(rt: EngineRuntime):
    text = random_text()
    pages = [len(text)]
    await rt.enqueue_high_priority_chunked_embedding_request(text, pages)

async def main_stress():
    with EngineRuntime(fd, size) as rt:
        embedding_completed = await pool(10, 60, lambda: embedding_task(rt))
        print(f"Embeddings Completed {embedding_completed}")

asyncio.run(main_stress())

