from typing import Any, Literal 
from dataclasses import dataclass
import struct, json

with open("vocab.txt", "rb") as f:
    vocab_data = f.read()

token_id = 0
prefix_dict = dict[Any, Any]()
d = prefix_dict
i = 0
col = 0
while i < len(vocab_data):
    b = vocab_data[i]
    if b == ord("[") and col == 0:
        token_id += 1
        i += 1
        while vocab_data[i] != ord("\n"):
            i += 1
        i += 1
        col = 0
    elif b == ord("\n"):
        d["id"] = token_id
        token_id += 1
        d = prefix_dict
        i += 1
        col = 0
    elif b == ord("#") and col == 0:
        c = b
        if i + 1 < len(vocab_data) and vocab_data[i + 1] == ord("#"):
            c = "##"
            i += 1
            col += 1
        if c not in d:
            d[c] = {}
        d = d[c]
        i += 1
        col += 1
    else:
        if b not in d:
            d[b] = {}
        d = d[b]
        i += 1
        col += 1

print(len(prefix_dict))

@dataclass
class TrieNode:
    type: tuple[Literal["root", "suffix_root"], int] | None
    id: int | None
    values: list[int]
    offsets: list[int]

def flatten(d: dict[Any, Any]) -> list[TrieNode]:
    r = list[TrieNode]()

    def imp(d: dict[Any, Any]) -> int:
        id = d.pop("id", None)
        values = list(d.keys())
        offsets = [0] * len(values)

        r.append(TrieNode(None, id, values, offsets))
        idx = len(r) - 1

        for i, v in enumerate(d.values()):
            r[idx].offsets[i] = len(r)
            imp(v)

        return idx 
    
    for k, v in d.items():
        if k == "##":
            for a, b in v.items():
                idx = imp(b)
                r[idx].type = ("suffix_root", a)
        else:
            idx = imp(v)
            r[idx].type = ("root", k)
    
    return r
    

flat_trie = flatten(prefix_dict)
print(len(flat_trie))

UNK = 100

def _trie_node_binary_blob(node: TrieNode, lookup: dict[int, int]) -> bytes:
    n = len(node.values)
    return struct.pack("<HB%dB%dQ" % (n, n), node.id if node.id is not None else UNK, n, *node.values, *(lookup[x] for x in node.offsets))

def _trie_node_binary_blob_size(node: TrieNode) -> int:
    return 2 + 1 + len(node.values) + len(node.values) * 8

def binary_offsets(trie: list[TrieNode]) -> dict[int, int]:
    cum = 0
    map = dict[int, int]()
    for i, node in enumerate(trie):
        size = _trie_node_binary_blob_size(node)
        map[i] = cum
        cum += size
    return map

def to_binary_blob(trie: list[TrieNode], map: dict[int, int]) -> bytes:
    data = bytes()
    for node in trie:
        data += _trie_node_binary_blob(node, map)
    return data

offset_map = binary_offsets(flat_trie)

binary_trie = to_binary_blob(flat_trie, offset_map)
print(len(binary_trie))

with open("trie.bin", "wb") as f:
    f.write(binary_trie)


def to_binary_root_blob(trie: list[TrieNode], map: dict[int, int]) -> bytes:
    data = [0xffffffffffffffff] * (256 * 2)
    for i, node in enumerate(trie):
        match node.type:
            case ("root", b):
                data[b] = map[i]
            case ("suffix_root", b):
                data[b + 256] = map[i]
            case _:
                pass
    return struct.pack("<%dQ" % (len(data),), *data)

binary_trie_root = to_binary_root_blob(flat_trie, offset_map) 
print(len(binary_trie_root))

with open("trie_root.bin", "wb") as f:
    f.write(binary_trie_root)


