# Improved word tokenizer (v2), a Unicode-aware fix for the scratch version.
#
# Changes from the string.punctuation approach in scratch/:
#   - Unicode punctuation (curly quotes, em dashes) separates words instead of
#     being left inside them, so "labs—no" is two words.
#   - URLs, decimal numbers and percentages stay whole (3.7% is not "37").
#   - Hyphenated words stay whole ("supply-chain").
#   - Symbols such as ≠ and ☕ are kept as their own tokens.
#   - Unknown words map to <unk> when encoding.
import re
import sys
import unicodedata
from pathlib import Path

TOKEN_RE = re.compile(
    r"https?://\S*[\w/]"  # URL (a trailing period or comma is left out)
    r"|\d+(?:\.\d+)?%?"  # number, decimal or percentage
    r"|\w+(?:[-']\w+)*"  # word, allowing inner hyphens/apostrophes
    r"|[^\w\s]"  # any other single character, filtered below
)


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'")  # curly apostrophe inside a word
    tokens = []
    for tok in TOKEN_RE.findall(text):
        if len(tok) == 1 and not tok.isalnum():
            # keep symbols (category S*), drop punctuation (category P*)
            if unicodedata.category(tok).startswith("S"):
                tokens.append(tok)
        else:
            tokens.append(tok)
    return tokens


def build_vocab(tokens):
    itos = ["<unk>"] + sorted(set(tokens))
    return {t: i for i, t in enumerate(itos)}, itos


def encode(tokens, stoi):
    return [stoi.get(t, 0) for t in tokens]


def decode(ids, itos):
    return " ".join(itos[i] for i in ids)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("labs—no") == ["labs", "no"]
    assert tokenize("rose 3.7%—blame") == ["rose", "3.7%", "blame"]
    assert tokenize("see https://a.org/b?c=4.") == ["see", "https://a.org/b?c=4"]
    assert tokenize("a ≠ b ☕") == ["a", "≠", "b", "☕"]
    assert tokenize("") == []

    text = (Path(__file__).with_name("input.txt")).read_text(encoding="utf-8")
    tokens = tokenize(text)
    stoi, itos = build_vocab(tokens)
    ids = encode(tokens, stoi)
    assert decode(ids, itos).split(" ") == tokens
    assert encode(["zzz"], stoi) == [0]
    print("Token count:", len(tokens))
    print("Vocab size (incl. <unk>):", len(itos))
    print(tokens)
