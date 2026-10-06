# Word-level tokenizer, version 1: the five scratch steps combined.
# Simple on purpose: only ASCII punctuation (string.punctuation) is removed.
import string
import sys
from pathlib import Path

INPUT = Path(__file__).with_name("input.txt")


# 1) lowercase, remove punctuation, split on whitespace
def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# 2) build the vocabulary
def build_vocab(tokens):
    itos = sorted(set(tokens))
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos


# 3) numericalize: tokens -> ids
def numericalize(tokens, stoi):
    return [stoi[t] for t in tokens]


# 4) decode: ids -> tokens
def decode(ids, itos):
    return [itos[i] for i in ids]


# 5) join tokens back into a sentence (punctuation and case are lost)
def join(tokens):
    return " ".join(tokens)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    text = INPUT.read_text(encoding="utf-8")
    tokens = tokenize(text)
    stoi, itos = build_vocab(tokens)
    ids = numericalize(tokens, stoi)
    assert decode(ids, itos) == tokens
    print("Input characters:", len(text))
    print("Token count after tokenization:", len(tokens))
    print("Vocab size:", len(itos))
    print(join(decode(ids, itos)))
