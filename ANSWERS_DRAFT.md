# HW4 answers (draft)

## Q3.1 Character tokenizer

Input: `tokenizers/input.txt` (341 characters, 374 UTF-8 bytes, no trailing newline).
Non-ASCII characters: `“ ” ‘ ’` (U+201C, U+201D, U+2018, U+2019), `—` (U+2014), `≠` (U+2260),
`é` (U+00E9), `☕` (U+2615). The ☕ is a single code point; there is **no** variation selector
(U+FE0F) after it in the saved file.

Lines in `tiny_rnn.py` that convert characters to ids and back:

| Line | Code | Role |
|---|---|---|
| 36 | `chars = sorted(list(set(text)))` | unique characters, sorted |
| 37 | `stoi = {c: i for i, c in enumerate(chars)}` | char -> id |
| 38 | `itos = {i: c for c, i in stoi.items()}` | id -> char |
| 39 | `vocab_size = len(chars)` | vocabulary size |
| 49 | `X_idx.append([stoi[c] for c in seq])` | encode input window |
| 50 | `y_idx.append(stoi[nxt])` | encode target |
| 109 | `char = itos[idx]` | decode a sampled id |

`tokenizers/char_tokenizer.py` uses the same approach. Round trip verified:
**341 tokens, vocabulary size 62.**

## Q3.2 Word tokenizer pseudocode

```
1. lowercase, remove punctuation, split
   text <- lowercase(text)
   text <- delete every character in string.punctuation
   words <- split text on whitespace
2. build the vocab
   itos <- sorted(unique(words))
   stoi <- map each word in itos to its index
3. numericalize
   ids <- [stoi[w] for each w in words]
4. decode
   words' <- [itos[i] for each i in ids]
5. join
   text' <- words' joined with single spaces
```

Each step is implemented and tested in `tokenizers/scratch/step1.py` ... `step5.py`.
Result on the input: 47 words, vocab 44, round trip exact for words but the joined text is
lossy (case, punctuation and line breaks are gone).

`string.punctuation` only covers ASCII punctuation, so it misses: `—` (em dash),
`“ ”` and `‘ ’` (curly quotes), `≠`, and `☕` (a symbol, not punctuation at all). Results:
`labs—no` stays one word, `3.7%—blame` becomes `37—blame`, and `“introtoai”` keeps its quotes.
It also deletes characters that matter: `3.7` becomes `37`, and the URL is mashed into
`httpsexampleorgabc42`.

## Q3.3 Tokenizer comparison

Subword tokens counted with `tiktoken`, encoding **`o200k_base`** (the one tiktoken maps
`gpt-5` and `gpt-4o` to): **111 tokens**, 341 / 111 = **3.07 characters per token**.

| Scheme | Tokens | Vocab (on this text) | Chars/token |
|---|---|---|---|
| Character | 341 | 62 | 1.00 |
| Word (scratch steps) | 47 | 44 | 7.26 |
| Word (`word_tokenizer_v2.py`) | 53 | 51 (incl. `<unk>`) | 6.43 |
| Subword (o200k_base) | 111 | n/a (about 200k fixed) | 3.07 |

Trade-offs:
- **Character:** tiny vocabulary and nothing is ever out-of-vocabulary, but sequences are long
  and each token carries little meaning, so the model must learn spelling and words from scratch.
- **Word:** short sequences and meaningful tokens, but the vocabulary is huge on real data,
  unseen words are out-of-vocabulary, and punctuation/case handling throws information away.
- **Subword (BPE):** a middle ground. Rare words split into pieces, so nothing is
  out-of-vocabulary, while common words stay as one token. The cost is a vocabulary that must be
  trained and tokens that are not always human-meaningful (`'/B'`, `'PE'`).

Why real-world tokenization is hard (examples from this text):
- Unicode: curly quotes, em dashes, `é` and emoji. The ☕ takes two tokens in o200k_base
  (it is split inside its UTF-8 bytes), and emoji can carry hidden variation selectors.
- Punctuation is meaningful sometimes (`3.7%`, a URL, `E.g.,`) and noise other times.
- Hyphenation and compounds: `intro-to-AI`, `supply-chain`, `ChatGPT-5`.
- Numbers, URLs and code have no natural word boundaries.
- Case: `Chat` vs `chat`; lowercasing loses information.
- Languages without spaces (Chinese, Japanese) cannot be split on whitespace.
- Tokenizers differ between models, so token counts and costs are not portable.

## Word tokenizer v2 (improved)

`tokenizers/word_tokenizer_v2.py` fixes the scratch version's Unicode problems: it treats
curly quotes and em dashes as separators (`labs—no` -> `labs`, `no`), keeps `3.7%`, hyphenated
words and the URL whole, keeps `≠` and `☕` as tokens, and maps unknown words to `<unk>`.
Result: **53 tokens, vocab 51.** Known weakness: `e.g.` splits into `e`, `g`.
