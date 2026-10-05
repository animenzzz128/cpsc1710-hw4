# Character-level tokenizer, built the same way as tiny_rnn.py (lines 36-39).
import sys
from pathlib import Path

text = Path(__file__).with_name("input.txt").read_text(encoding="utf-8")

chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


if __name__ == "__main__":
    ids = encode(text)
    assert decode(ids) == text, "round trip failed"
    print("Round trip OK")
    print("Token count:", len(ids))
    print("Vocab size:", vocab_size)
    print("Vocab:", ascii("".join(chars)))
