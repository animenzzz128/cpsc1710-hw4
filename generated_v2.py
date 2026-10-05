import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 1710
if SEED is not None:
    tf.keras.utils.set_random_seed(SEED)  # seeds python, numpy, tensorflow
    np.random.seed(SEED)
rng = np.random.default_rng(SEED)

SEQ_LEN = 40
sentences = [
    "I like cats.", "I like dogs.", "I like noodles.", "I like tacos.",
    "I like books.", "I like robots.", "I like music.", "I like puzzles.",
    "I like pizza.", "I like coding.",
]
text = "\n".join(sentences)
print("Corpus length:", len(text))

vocab = sorted(set(text))
char_to_id = {c: i for i, c in enumerate(vocab)}
id_to_char = {i: c for i, c in enumerate(vocab)}
print("Vocab:", vocab)

X, y = [], []
for i in range(len(text) - SEQ_LEN):
    X.append([char_to_id[c] for c in text[i:i + SEQ_LEN]])
    y.append(char_to_id[text[i + SEQ_LEN]])
X = np.array(X, dtype=np.int32)
y = np.array(y, dtype=np.int32)
print("Num training samples:", len(X))

model = keras.Sequential([
    layers.Embedding(len(vocab), 32),
    layers.LSTM(128, return_sequences=False),
    layers.Dense(len(vocab)),
])
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
)
hist = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
print("Final loss:", float(hist.history["loss"][-1]))


def next_logits(window_ids):
    arr = np.array([window_ids], dtype=np.int32)
    return model(arr, training=False).numpy()[0]


def probs_from_logits(logits, temperature):
    z = logits / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_window(s):
    s = s.rjust(SEQ_LEN)[-SEQ_LEN:]
    return [char_to_id[c] for c in s]


def generate(seed_text, length, temperature):
    window = make_window(seed_text)
    out = list(seed_text.rjust(SEQ_LEN)[-SEQ_LEN:])
    for _ in range(length):
        logits = next_logits(window)
        if temperature <= 0:
            nid = int(np.argmax(logits))
        else:
            p = probs_from_logits(logits, temperature)
            nid = int(rng.choice(len(p), p=p))
        out.append(id_to_char[nid])
        window = window[1:] + [nid]
    return "".join(out)


PROMPT = "I like "
for t in [0.1, 0.5, 0.7, 1.0]:
    print(f"\n=== Temperature {t} ===")
    print("Top probabilities for the first generated character:")
    p = probs_from_logits(next_logits(make_window(PROMPT)), t)
    for idx in np.argsort(p)[::-1][:5]:
        print(f"  {id_to_char[int(idx)]!r}: {p[idx] * 100:.1f}%")
    print()
    print(generate(PROMPT, 180, t))
