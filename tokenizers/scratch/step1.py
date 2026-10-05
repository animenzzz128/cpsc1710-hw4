# Step 1: lowercase, remove punctuation, split into words.
import string
import sys
from pathlib import Path

INPUT = Path(__file__).resolve().parent.parent / "input.txt"


def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("a-b  c\nd") == ["ab", "c", "d"]
    assert tokenize("") == []
    words = tokenize(INPUT.read_text(encoding="utf-8"))
    print("Word count:", len(words))
    print(words)
    leftover = sorted({c for w in words for c in w if not c.isalnum()})
    print("Punctuation-like characters string.punctuation missed:")
    for c in leftover:
        print(f"  {c!r} U+{ord(c):04X}")
