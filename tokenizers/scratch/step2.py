# Step 2: build the vocabulary (word -> id and id -> word).
import sys
from pathlib import Path

from step1 import INPUT, tokenize


def build_vocab(words):
    itos = sorted(set(words))
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    stoi, itos = build_vocab(["b", "a", "b"])
    assert stoi == {"a": 0, "b": 1} and itos == ["a", "b"]
    words = tokenize(INPUT.read_text(encoding="utf-8"))
    stoi, itos = build_vocab(words)
    assert all(itos[stoi[w]] == w for w in words)
    print("Vocab size:", len(stoi))
    print(itos)
