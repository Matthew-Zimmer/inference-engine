import dataclasses, time, sys, random
from subprocess import Popen, PIPE
from typing import Callable


chars = "qwertyuiopasdfghjklzxcvbnm0123456789"
whitespace = " \t\n\r"

def random_slop(n: int):
    return "".join(random.choice(chars) for _ in range(n))

def random_whitespace():
    n = random.randint(1, 10)
    return "".join(random.choice(whitespace) for _ in range(n))

def random_slop_sentence(n: int):
    text = ""
    for _ in range(n):
        text += random_slop(random.randint(1, 16))
        text += random_whitespace()
    return text

with open("words.txt") as f:
    words = f.read().splitlines()

def random_sentence(n: int) -> str:
    return " ".join(random.choice(words) for _ in range(n)).lower()

def compare_embeddings(l: list[float], r: list[float], threshold: float) -> bool:
    return len(l) == len(r) and all(abs(x - y) < threshold for x, y in zip(l, r))


@dataclasses.dataclass
class ModelStats:
    tokens: list[int]
    tokenization_time_us: float
    embeddings: list[float]
    embedding_time_us: float

    @staticmethod
    def from_stdout(text: str) -> "ModelStats":
        lines = text.splitlines()
        tokens = [int(x) for x in lines[0].split(",")]
        tokenization_time_us = float(lines[1])
        embeddings = [float(x) for x in lines[2].split(",")]
        embedding_time_us = float(lines[3])
        return ModelStats(tokens, tokenization_time_us, embeddings, embedding_time_us)


def compare(text: str, threshold: float):
    print(f"Comparing HF vs IE (threshold: {threshold}) for input text '{text}'")

    print("Running HF code")
    hf = Popen(["python", "hf.py", text], stdout=PIPE)
    hf.wait()
    if hf.stdout is None: return False 
    print("Running HF code done reading stdout")
    hf_out = hf.stdout.read().decode()
    hf_stats = ModelStats.from_stdout(hf_out)
    
    print("Running IE code")
    me = Popen(["../zig-out/bin/inference-engine", text], stdout=PIPE)
    time.sleep(3)
    me.terminate()
    if me.stdout is None: return False
    print("Running IE code done reading stdout")
    me_out = me.stdout.read().decode()
    me_stats = ModelStats.from_stdout(me_out)

    if me_stats.tokens != hf_stats.tokens:
        print("Tokens do not match")
        print(f"HF: {hf_stats.tokens}")
        print(f"IE: {me_stats.tokens}")
        return False
    
    if not compare_embeddings(me_stats.embeddings, hf_stats.embeddings, threshold):
        print("Emebddings do not match")
        print(f"HF: {hf_stats.embeddings[:5]}...")
        print(f"IE: {me_stats.embeddings[:5]}...")
        return False

    print("tokens and embedddings match!")
    print(f"HF: tokenization took: {hf_stats.tokenization_time_us:.02f} us")
    print(f"IE: tokenization took: {me_stats.tokenization_time_us:.02f} us")
    
    print(f"HF: emebdding took: {hf_stats.embedding_time_us:.02f} us")
    print(f"IE: embedding took: {me_stats.embedding_time_us:.02f} us")

    return True

def at[T, R](l: list[T], i: int, map: Callable[[T], R]) -> R | None:
    if 0 <= i < len(l):
        return map(l[i])
    return None

def main(args: list[str]):
    threshold = at(args, 0, float) or 5e-6
    mode = at(args, 1, int) or 0
    good = True

    match mode:
        case 0:
            text = at(args, 2, str) or ''
            good &= compare(text, threshold)
        case 1:
            samples = at(args, 2, int) or 100
            for _ in range(samples):
                good &= compare(random_sentence(random.randint(100, 500)), threshold)
        case 2:
            samples = at(args, 2, int) or 100
            for _ in range(samples):
                good &= compare(random_slop_sentence(random.randint(75, 250)), threshold)
        case _:
            print("unknown compare mode valid options: 0,1,2")

    if not good:
        exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])


