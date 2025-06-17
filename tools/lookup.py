import sys

with open("../vocab.txt") as f:
    tokens = {i: x for i, x in enumerate(f.read().splitlines())}

ids = [int(x) for x in sys.argv[1].split(", ")]

print(" ".join(tokens[id] for id in ids))

