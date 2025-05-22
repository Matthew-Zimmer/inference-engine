from typing import Any, Literal 
from dataclasses import dataclass
import struct

with open("vocab.txt", "rb") as f:
    vocab_data = f.read()

token_id = 0
prefix_dict = dict[Any, Any]()
d = prefix_dict
i = 0
while i < len(vocab_data):
    b = vocab_data[i]
    if b == ord("\n"):
        d["id"] = token_id
        token_id += 1
        d = prefix_dict
        i += 1
    elif b == ord("#"):
        c = b
        if i + 1 < len(vocab_data) and vocab_data[i + 1] == ord("#"):
            c = "##"
            i += 1
        if c not in d:
            d[c] = {}
        d = d[c]
        i += 1
    else:
        if b not in d:
            d[b] = {}
        d = d[b]
        i += 1

print(len(prefix_dict))
prefix_dict.pop("id", None)


@dataclass
class TrieNode:
    type: tuple[Literal["root", "suffix_root"], int] | None
    id: int | None
    values: list[int]
    offsets: list[int]

def flatten(d: dict[Any, Any]) -> list[TrieNode]:
    def imp(d: dict[Any, Any]) -> list[TrieNode]:
        id = d.pop("id", None)
        values = list(d.keys())
        offsets = [0] * len(values)

        r = [TrieNode(None, id, values, offsets)]

        for i, v in enumerate(d.values()):
            r[0].offsets[i] = len(r)
            r.extend(imp(v))

        return r
    
    r = list[TrieNode]()
    for k, v in d.items():
        # print(k, v)
        if k == "##":
            for a, b in v.items():
                l = imp(b)
                l[0].type = ("suffix_root", a)
                r.extend(l)
        else:
            l = imp(v)
            l[0].type = ("root", k)
            r.extend(l)
    return r
    

flat_trie = flatten(prefix_dict)
print(len(flat_trie))

UNK = 101

def _trie_node_binary_blob(node: TrieNode, lookup: dict[int, int]) -> bytes:
    n = len(node.values)
    return struct.pack("<IB%dB%dQ" % (n, n), node.id or UNK, n, *node.values, *(lookup[x] for x in node.offsets))

def _trie_node_binary_blob_size(node: TrieNode) -> int:
    return 4 + 1 + len(node.values) + len(node.values) * 8

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


