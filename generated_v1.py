import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 1710
if SEED is not None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)

SEQ_LEN = 40

sentences = [
    "I like cats.", "I like dogs.", "I like noodles.", "I like tacos.",
    "I like books.", "I like robots.", "I like music.", "I like puzzles.",
    "I like pizza.", "I like coding.",
]
text = "\n".join(sentences)
print("Corpus length:", len(text))

vocab = sorted(set(text))
char2id = {c: i for i, c in enumerate(vocab)}
id2char = {i: c for c, i in char2id.items()}
vocab_size = len(vocab)
print("Vocabulary:", vocab)

ids = np.array([char2id[c] for c in text], dtype=np.int32)
X = np.array([ids[i:i + SEQ_LEN] for i in range(len(ids) - SEQ_LEN)], dtype=np.int32)
y = np.array([ids[i + SEQ_LEN] for i in range(len(ids) - SEQ_LEN)], dtype=np.int32)
print("Num training samples:", len(X))

model = keras.Sequential([
    layers.Embedding(vocab_size, 32),
    layers.LSTM(128, return_sequences=False),
    layers.Dense(vocab_size),
])
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
)
history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
print("Final loss:", history.history["loss"][-1])


def next_logits(window_ids):
    x = np.array([window_ids], dtype=np.int32)
    return model(x, training=False).numpy()[0]


def softmax_t(logits, temperature):
    z = logits / temperature
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def generate(seed_text, length, temperature, rng):
    s = seed_text.rjust(SEQ_LEN)
    for _ in range(length):
        window = [char2id[c] for c in s[-SEQ_LEN:]]
        logits = next_logits(window)
        if temperature <= 0:
            nid = int(np.argmax(logits))
        else:
            p = softmax_t(logits, temperature)
            nid = int(rng.choice(len(p), p=p))
        s += id2char[nid]
    return s


rng = np.random.default_rng(SEED)
prompt = "I like "
for t in [0.1, 0.5, 0.7, 1.0]:
    print(f"\n=== Temperature {t} ===")
    print("Top probabilities for the first generated character:")
    window = [char2id[c] for c in prompt.rjust(SEQ_LEN)]
    p = softmax_t(next_logits(window), t)
    for i in np.argsort(p)[::-1][:5]:
        print(f"  {id2char[int(i)]!r}: {p[i] * 100:.1f}%")
    print(generate(prompt, 180, t, rng))
