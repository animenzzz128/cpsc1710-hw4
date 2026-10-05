import unicodedata
from pathlib import Path

text = Path(__file__).with_name("input.txt").read_text(encoding="utf-8")
print("characters:", len(text), "| bytes:", len(text.encode("utf-8")))
print("ends with newline:", text.endswith("\n"))
for i, c in enumerate(text):
    if ord(c) > 127:
        print(f"index {i:3d}  U+{ord(c):04X}  {unicodedata.name(c, '?')}")
i = text.index("☕")
print("after coffee:", ascii(text[i : i + 2]), "-> variation selector present:", "️" in text)
