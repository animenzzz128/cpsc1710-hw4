# Tiny RNN text generator
#
# Big picture: we show a small neural network ten short sentences, teach it to
# guess "given the last 40 characters, what character comes next?", then let it
# write new text by repeatedly guessing the next character and feeding it back in.

# `import X as Y` loads a library and gives it a short nickname.
import numpy as np  # NumPy: fast arrays and math on whole arrays at once
import tensorflow as tf  # TensorFlow: the deep-learning library (Keras lives inside it)


# ---------- 0) Reproducibility ----------
# Keep the first run reproducible so classmates can compare results.
# Change this number—or set it to None—to experiment with different runs.
SEED = 1710  # ALL_CAPS is a Python convention for "constant, don't change while running"
if SEED is not None:  # `is None` / `is not None` is the idiomatic way to test for None
    # Seeds Python's, NumPy's and TensorFlow's random generators in one call, so the
    # model's starting weights (random!) are the same every run.
    tf.keras.utils.set_random_seed(SEED)
# A separate NumPy random generator, used later to pick characters while sampling.
# default_rng(None) means "seed from the OS", i.e. different every run.
rng = np.random.default_rng(SEED)


# ---------- 1) Tiny corpus (swap this block to change tasks) ----------
# The "corpus" is the training text. Everything the model can ever say comes from it.
tiny_lines = [
    "I like cats.",
    "I like dogs.",
    "I like noodles.",
    "I like tacos.",
    "I like books.",
    "I like robots.",
    "I like music.",
    "I like puzzles.",
    "I like pizza.",
    "I like coding.",
]
# "\n".join(list) glues the strings together with a newline between each, giving one
# long string. The newline is a character the model learns, so it learns where
# sentences end.
text = "\n".join(tiny_lines)

# Alternative corpora (uncomment one):
# DNA: text = "TATAAA\nCGCGCG\nATG...TAA\nACGTACGTACGT\n"
# Emoji: text = "☀️🌤️⛅🌧️⛈️🌈\n🍞🧈🍯\n🥚🍳🍞\n🙂➡️😊\n"
# Nursery: text = "Twinkle twinkle little star,\nHow I wonder what you are.\n"

print("Corpus length:", len(text))
# Neural networks work on numbers, not letters, so we build a lookup table.
# set(text) -> the unique characters; sorted(...) -> a stable alphabetical order
# (so each character always gets the same number); list(...) -> make it a list.
chars = sorted(list(set(text)))
# "string to int": {character: number}. This is a *dict comprehension*.
# enumerate(chars) yields pairs (0, first_char), (1, second_char), ...
stoi = {c: i for i, c in enumerate(chars)}
# "int to string": the reverse table. `.items()` yields (key, value) pairs.
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)  # how many distinct characters the model must choose between
print("Vocab:", chars)

# ---------- 2) Vectorize to (input sequence -> next char) pairs ----------
# Turn the text into many training examples using a sliding window:
# input = 40 characters in a row, answer = the 41st character.
seq_len = 40  # how many characters of context the model sees
step = 1  # slide the window one character at a time
X_idx, y_idx = [], []  # inputs and answers; `a, b = [], []` makes two empty lists
# range(start, stop, step): the last window must still have a next character after it,
# hence stop = len(text) - seq_len. This gives 143 - 40 = 103 examples.
for i in range(0, len(text) - seq_len, step):
    seq = text[i : i + seq_len]  # slicing: characters i up to (not including) i+40
    nxt = text[i + seq_len]  # the single character right after the window
    # List comprehension: build a list by converting each character to its number.
    X_idx.append([stoi[c] for c in seq])
    y_idx.append(stoi[nxt])

# Convert Python lists to NumPy arrays, which is what Keras expects.
# X has shape (103, 40): 103 examples x 40 character-numbers each.
# y has shape (103,): one correct next-character number per example.
# int32 = 32-bit integers (these are lookup indexes, not decimals).
X = np.array(X_idx, dtype=np.int32)
y = np.array(y_idx, dtype=np.int32)
print("Num training samples:", len(X))

# ---------- 3) Model ----------
# Sequential = a simple stack of layers; data flows through them top to bottom.
model = tf.keras.Sequential(
    [
        # Embedding: replaces each character number with a learned list of 32
        # numbers (a "vector"). Similar characters can end up with similar vectors.
        # vocab_size is how many rows the lookup table has.
        # Shape: (batch, 40) -> (batch, 40, 32)
        tf.keras.layers.Embedding(vocab_size, 32),
        # LSTM (Long Short-Term Memory) is the recurrent part. It reads the 40
        # vectors one at a time, in order, carrying a 128-number "memory" forward
        # and deciding at each step what to remember or forget. It returns only
        # the final memory: (batch, 40, 32) -> (batch, 128).
        tf.keras.layers.LSTM(128),
        # Dense: a fully connected layer that turns the 128-number summary into
        # one score per vocabulary character: (batch, 128) -> (batch, vocab_size).
        # These raw scores are called "logits" (not yet probabilities).
        tf.keras.layers.Dense(vocab_size),
    ]
)
# compile() chooses how learning happens (it doesn't train yet).
model.compile(
    # Adam: a popular optimizer, i.e. the rule for nudging weights to reduce
    # loss. 1e-2 is scientific notation for 0.01, the learning rate (step size).
    optimizer=tf.keras.optimizers.Adam(1e-2),
    # Loss = how wrong the model is; training tries to shrink it.
    # "Sparse" = the answers are plain integers (e.g. 7), not one-hot vectors.
    # "Categorical crossentropy" = penalize low probability on the right answer.
    # from_logits=True = our last layer outputs raw scores, so the loss should
    # apply softmax itself.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# ---------- 4) Train (tiny & fast) ----------
# fit() is the training loop. Per epoch (one full pass over the 103 examples),
# the data is processed in batches of 64 (so 2 weight updates per epoch), 20 epochs.
# verbose=0 silences the progress bar. It returns a History object.
history = model.fit(X, y, batch_size=64, epochs=20, verbose=0)
# history.history is a dict of lists, e.g. {"loss": [one value per epoch]}.
# [-1] means "last item" -> the loss after the final epoch.
print("Final loss:", history.history["loss"][-1])


# ---------- 5) Sampling helper with temperature ----------
# `def` defines a function. `temperature=1.0` is a default argument value.
def sample_logits(logits, temperature=1.0):
    # Pick one character index from the model's raw scores.
    if temperature <= 0:  # greedy: always take the highest score, no randomness
        return int(np.argmax(logits))  # argmax = position of the largest value
    # Temperature: dividing by a number < 1 exaggerates differences (more
    # predictable); dividing by > 1 flattens them (more random).
    logits = logits / temperature
    # Softmax turns scores into probabilities that are positive and sum to 1.
    # .numpy() converts a TensorFlow tensor to a NumPy array.
    probabilities = tf.nn.softmax(logits).numpy()
    # Roll a weighted die: choose an index from 0..vocab_size-1 where p gives
    # each index's chance. int(...) converts NumPy's integer type to a plain int.
    return int(rng.choice(len(probabilities), p=probabilities))


def next_char_probabilities(seed="I like ", temperature=1.0, top_n=5):
    """Return the most likely next characters for a prompt."""
    # Conditional expression: `A if condition else B`. If the seed is shorter than
    # 40 characters, pad the left with spaces (" " * n repeats the string n times)
    # because the model always expects exactly 40 characters.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed)
    # seed[-seq_len:] = the last 40 characters (negative index counts from the end).
    # dict.get(key, default): look up the character, or use 0 if it was never in
    # the training text (so unknown characters don't crash it).
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    # Wrap in a list so the shape is (1, 40): a batch containing one example.
    x = np.array([context], dtype=np.int32)
    # predict() runs the model forward. It returns shape (1, vocab_size);
    # [0] picks the single row, giving one score per character.
    logits = model.predict(x, verbose=0)[0]

    if temperature <= 0:
        # Greedy: put 100% probability on the top character, 0 on the rest.
        # zeros_like makes an all-zero array with the same shape as logits.
        probabilities = np.zeros_like(logits, dtype=np.float64)
        probabilities[np.argmax(logits)] = 1.0
    else:
        probabilities = tf.nn.softmax(logits / temperature).numpy()

    # argsort gives indexes ordered smallest -> largest probability.
    # [-top_n:] takes the last 5 (the largest); [::-1] reverses so largest is first.
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    # Return a list of (character, probability) pairs. itos converts the index back
    # to a letter; int()/float() convert NumPy types to plain Python types.
    return [(itos[int(i)], float(probabilities[i])) for i in top_indices]


def generate(seed="I like ", n_chars=200, temperature=0.7):
    # Write text by repeatedly predicting the next character.
    # Ensure seed length is at least seq_len by left-padding with spaces.
    seed = seed if len(seed) >= seq_len else (" " * (seq_len - len(seed)) + seed)
    context = [stoi.get(c, 0) for c in seed[-seq_len:]]
    output = list(seed)  # list("abc") -> ["a", "b", "c"]; we append to this
    # `_` is the conventional name for a loop variable we never use.
    for _ in range(n_chars):
        x = np.array([context], dtype=np.int32)
        logits = model.predict(x, verbose=0)[0]
        idx = sample_logits(logits, temperature)  # choose the next character's index
        char = itos[idx]
        output.append(char)
        # Slide the window: drop the oldest character, add the new one, so the
        # model's own output becomes part of its next input. This feedback loop is
        # what "autoregressive" generation means.
        context = context[1:] + [idx]
    # "".join(list_of_chars) glues characters back into one string. Note the output
    # includes the padded seed, which is why printed samples start with blank space.
    return "".join(output)


# ---------- 6) Try a few temperatures ----------
# Same model, same prompt; only the randomness changes.
for temperature in [0.1, 0.5, 0.7, 1.0]:
    print("\n=== Temperature", temperature, "===")  # "\n" = blank line first
    print("Top probabilities for the first generated character:")
    # Each returned item is a (char, probability) pair; the for-loop *unpacks* it
    # into two variables.
    for char, probability in next_char_probabilities(
        seed="I like ", temperature=temperature
    ):
        # f-string: {expr} is replaced by its value. `!r` shows the repr (quotes,
        # and "\n" shown as \n); `:.1%` formats 0.998 as "99.8%".
        print(f"  {char!r}: {probability:.1%}")
    print(generate(seed="I like ", n_chars=180, temperature=temperature))
