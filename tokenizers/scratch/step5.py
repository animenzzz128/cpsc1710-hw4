# Step 5: join decoded words back into a string.
# The result is NOT the original text: case, punctuation and spacing are gone.
import sys

from step1 import INPUT, tokenize
from step2 import build_vocab
from step3 import numericalize
from step4 import decode


def join(words):
    return " ".join(words)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    assert join(["a", "b"]) == "a b"
    assert join([]) == ""
    text = INPUT.read_text(encoding="utf-8")
    words = tokenize(text)
    stoi, itos = build_vocab(words)
    rebuilt = join(decode(numericalize(words, stoi), itos))
    assert rebuilt == join(words)
    assert rebuilt != text  # lossy
    print(rebuilt)
