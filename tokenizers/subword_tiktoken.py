# Q3.3: count subword (BPE) tokens with tiktoken.
import sys
from pathlib import Path

import tiktoken

sys.stdout.reconfigure(encoding="utf-8")
text = Path(__file__).with_name("input.txt").read_text(encoding="utf-8")

for model in ["gpt-5", "gpt-4o"]:
    try:
        print(model, "->", tiktoken.encoding_for_model(model).name)
    except KeyError:
        print(model, "-> not known to this tiktoken version")

enc = tiktoken.get_encoding("o200k_base")
ids = enc.encode(text)
assert enc.decode(ids) == text
print("Encoding:", enc.name)
print("Characters:", len(text))
print("Tokens:", len(ids))
print("Characters per token: %.2f" % (len(text) / len(ids)))
print([enc.decode([i]) for i in ids])
