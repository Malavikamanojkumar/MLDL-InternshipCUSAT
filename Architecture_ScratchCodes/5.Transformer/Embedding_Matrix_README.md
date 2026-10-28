# Embedding Matrix & Positional Encoding — From Scratch

A NumPy-only walkthrough of how raw text becomes the numeric input a Transformer
actually operates on: tokenization → embedding lookup → embedding matrix →
sinusoidal positional encoding. This is the final module in the architectures
series — it stops short of a full Transformer (no attention, no training loop),
and instead focuses on getting the **input representation** exactly right, since
that's the piece every attention-based architecture is built on top of.

## Builds On

[`LSTM — From Scratch`](../lstm/README.md) processed sequences of *numeric*
readings. Text isn't numeric to begin with, so this module answers a different
question first: how do you turn a sequence of *words* into a sequence of vectors
a network can process — while still preserving the order they appeared in?

## Scope

Unlike the other modules, this one is **not trainable** — there's no loss, no
backward pass, no weight updates. The embedding vectors here are randomly
initialized and left as-is; in a real model they'd be learned end-to-end from a
downstream task's loss (or loaded pretrained). The goal of this module is purely
to make the *mechanics* of the representation layer visible:

1. Map words to integer token IDs
2. Look up (or randomly initialize) a feature vector per token ID
3. Stack those vectors into an embedding matrix
4. Compute a positional encoding for each position in a sequence
5. Combine word embedding + positional encoding into one position-aware vector

## Math Intuition

**Token → embedding lookup**: each unique word gets an integer ID, and each ID
indexes into a matrix of learned (or here, random) vectors:

```
embedding_matrix: shape (vocab_size, embedding_dim)
embedding_matrix[token_id]  →  the d-dimensional vector for that word
```

There's no computation here — it's a lookup table. The matrix itself is exactly
what an embedding layer in a real network learns during training.

**Positional encoding**: because looking up embeddings doesn't encode *order* at
all (attention, when added later, has no built-in sense of sequence position
either), a fixed sinusoidal pattern is added per position:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))     # even dimensions
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))     # odd dimensions
```

Using sine/cosine at different frequencies per dimension means each position
gets a unique pattern, and — importantly — the *relative* offset between any two
positions is encoded consistently regardless of absolute position, which is what
lets a Transformer generalize to sequence lengths it wasn't trained on.

**Combining**: word embedding and positional encoding are simply added,
elementwise, giving each word a vector that encodes both *what it is* and
*where it is*:

```
final_embedding = word_embedding + positional_encoding
```

## Implementation Notes

- Vocabulary is built from a small hardcoded word list; token IDs are assigned
  alphabetically for determinism.
- Embedding vectors are drawn uniformly from `[-1, 1]` — a stand-in for what
  would otherwise be a learned or pretrained embedding matrix.
- `positional_encoding(position, d_model)` computes one position's encoding
  vector using the sin/cos formula above; it's called once per position in the
  sample sentence to build the full positional-encoding matrix.
- Uses `display()` (Jupyter/IPython-only) for the embedding tables — swap for
  `print(df.round(3))` if running as a plain `.py` script outside a notebook.

## How to Run

**Dependencies**: `numpy`, `pandas`

```bash
pip install numpy pandas
```

Run top to bottom in a Jupyter notebook (for the `display()` calls). It prints
the vocabulary and token IDs, each word's embedding, the full embedding matrix,
the positional encoding for a sample sentence, and the final combined
(position-aware) embeddings.

## What's Next

This is the last module in the **architectures** folder. The companion
**linear algebra** folder (CR, QR, LU decomposition, Hessian, SVD, PCA, etc.)
covers the mathematical foundations these architectures are built on.
