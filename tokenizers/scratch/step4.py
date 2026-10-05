# Step 4: decode (ids -> words).
import sys

from step1 import INPUT, tokenize
from step2 import build_vocab
from step3 import numericalize


def decode(ids, itos):
    return [itos[i] for i in ids]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert decode([1, 0], ["a", "b"]) == ["b", "a"]
    words = tokenize(INPUT.read_text(encoding="utf-8"))
    stoi, itos = build_vocab(words)
    decoded = decode(numericalize(words, stoi), itos)
    assert decoded == words
    print("Decoded words match original words:", len(decoded))
