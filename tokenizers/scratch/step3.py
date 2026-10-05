# Step 3: numericalize (words -> ids).
import sys

from step1 import INPUT, tokenize
from step2 import build_vocab


def numericalize(words, stoi):
    return [stoi[w] for w in words]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert numericalize(["b", "a"], {"a": 0, "b": 1}) == [1, 0]
    words = tokenize(INPUT.read_text(encoding="utf-8"))
    stoi, _ = build_vocab(words)
    ids = numericalize(words, stoi)
    assert len(ids) == len(words)
    print("Token count:", len(ids))
    print(ids)
