# CR Decomposition

## Concept

CR decomposition factors a matrix `A` into `C` — a subset of A's own columns —
and `R`, a small coefficient matrix that reconstructs *every* column of `A`
(including the ones left out of `C`) as a linear combination of `C`'s columns:
`A = C · R`. `C` contains only the linearly independent columns of `A`; `R`
holds the "recipe" for rebuilding everything else from them.

## Why It's Used in ML

Real datasets often contain features that are exact (or near-exact) linear
combinations of other features — e.g. a `Total_Cost` column that's really just
`Item_Price + Shipping_Fee` added together. Such columns add no new information
and can hurt some models. CR decomposition gives a principled way to detect and
remove them: columns that end up in `C` are genuinely independent; anything
left out is provably reconstructible from `C` alone, so dropping it loses
nothing.

## The Math

If `A` has rank `r`, only `r` of its columns are truly independent:

```
A = C · R
```

- `C` = the `r` pivot columns of `A`, taken directly from `A` (not from its RREF)
- `R` = the first `r` rows of `A`'s row-reduced echelon form (RREF)

Each row of `R` corresponds to one column of `C`; each column of `R` gives the
weights that reconstruct that column of `A`. A pivot column's own weights form
an identity pattern (100% itself); a redundant column's weights show exactly
which combination of `C`'s columns produced it.

## Step-by-Step Walkthrough

1. **Build a dataset with a known redundancy** — 4 features (`Item_Price`,
   `Shipping_Fee`, `Total_Cost`, `Quantity`), where `Total_Cost` is
   deliberately set to `Item_Price + Shipping_Fee`, so the "correct" answer is
   known in advance.
2. **Compute the RREF of `A`** using `sympy.Matrix.rref()`, which returns both
   the RREF matrix and the pivot column indices — exact symbolic arithmetic
   avoids floating-point pivot-detection errors.
3. **Construct `C`** by indexing `A` at the pivot columns directly — not by
   taking columns from the RREF (whose pivot columns are just identity
   vectors, not the original data).
4. **Identify redundant columns** as every column index *not* in the pivot set.
5. **Verify the decomposition** — trim `R` to its first `rank` rows (the RREF
   pads the rest with all-zero rows) and check `C @ R_trimmed` reproduces `A`
   exactly.
6. **Drop the redundant columns**, keeping only the independent features.

## How to Run

**Dependencies**: `numpy`, `sympy`

```bash
pip install numpy sympy
```

Run top to bottom. It prints the original matrix, the discovered `C` and `R`,
which feature was flagged as redundant and why, the `C @ R` verification
against the original `A`, and the final reduced dataset.
