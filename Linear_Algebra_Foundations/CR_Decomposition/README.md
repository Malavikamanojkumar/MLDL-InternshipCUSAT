# CR Decomposition

A matrix full of redundant columns can be written as a **product of two smaller matrices**:

$$
A_{m\times n} = C_{m\times r}R_{r\times n}, \qquad r = \text{rank}(A)
$$

- **$C$** holds the *pieces*: the $r$ linearly independent columns of $A$, copied exactly as they appear in $A$.
- **$R$** holds the *instructions*: the non-zero rows of the reduced row echelon form (RREF) of $A$. It says how to put the pieces back together to get every column of $A$.

The decomposition is usually credited to Gilbert Strang. It is the most elementary of the standard factorisations, since it needs nothing beyond row reduction.

---

## Illustration

Take a $3\times 6$ matrix with a zero column and a few dependent columns:

$$
A=\begin{pmatrix}
1 & 3 & 5 & 15 & 0 & e\\
2 & 6 & 4 & 24 & 0 & \pi\\
4 & 12 & 1 & 41 & 0 & \sqrt3
\end{pmatrix}
$$

Row-reducing $A$ gives

$$
R=\begin{pmatrix}
1 & 3 & 0 & 10 & 0 & 0\\
0 & 0 & 1 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

The pivots sit in columns 1, 3 and 6. Those columns of the **original** $A$ become $C$:

$$
C=\begin{pmatrix}
1 & 5 & e\\
2 & 4 & \pi\\
4 & 1 & \sqrt3
\end{pmatrix}
$$

and $A = CR$.

### Reading $R$ one column at a time

Each column of $R$ is a *column picker*. It says which columns of $C$ to take, and in what amounts, to produce the matching column of $A$:

| Column of $R$ | Instruction | Column of $A$ it produces |
|---|---|---|
| $R_{c1}$ | take $1\cdot c_1$ | $a_1$ |
| $R_{c2}$ | take $3\cdot c_1$ | $a_2$ |
| $R_{c3}$ | take $1\cdot c_2$ | $a_3$ |
| $R_{c4}$ | take $10\,c_1 + c_2$ | $a_4$ |
| $R_{c5}$ | take nothing | $a_5 = \mathbf 0$ |
| $R_{c6}$ | take $1\cdot c_3$ | $a_6$ |

The pivot columns of $R$ ($R_{c1}, R_{c3}, R_{c6}$) are just the identity: "copy this column of $C$ unchanged". Every non-pivot column of $R$ is a genuine recipe. Column 4, for instance, says $a_4 = 10\,a_1 + a_3$, which you can check: $10(1,2,4)+(5,4,1) = (15,24,41)$.

So $R$ always contains an $r\times r$ identity block (in the pivot columns), and its remaining columns record the dependencies among the columns of $A$.

---

## What $C$ and $R$ tells us about $A$?

**Column space.** $C$ spans exactly the same space as $A$, so $\operatorname{col}(C)=\operatorname{col}(A)$ and $\dim = r$. Because of this, whether $A\mathbf x=\mathbf b$ is solvable can be checked against $C$ alone: a solution exists iff $\mathbf b\in\operatorname{col}(C)$.

**Row space.** Every row of $A$ is a combination of the $r$ rows of $R$, so $R$ is a clean basis for the row space of $A$. Unlike $C$, these rows are *not* rows of the original matrix.

**Null space.** Since the columns of $C$ are independent, $C(\cdot)=0$ forces $(\cdot)=0$. Therefore

$$
A\mathbf x = CR\mathbf x = 0 \;\Longrightarrow\; R\mathbf x = 0,
$$

so $\mathcal N(A)=\mathcal N(R)$. The directions that $A$ throws away are exactly the ones $R$ throws away.

**Rank.** Both factors have rank $r$, and $\dim\operatorname{col}(A)=\dim\operatorname{row}(A)=r$.

Another way to read the product is as a sum of rank-one pieces: if $C$ has columns $c_k$ and $R$ has rows $\rho_k^\top$, then

$$
A=\sum_{k=1}^{r} c_k\,\rho_k^\top .
$$

---

## Why use CR rather than SVD or PCA?

SVD and PCA also compress a matrix, but they build **new axes**, each a mixture of the original features. For a store dataset, a principal component might look like

$$
0.69\,(\text{Rice}) + 0.71\,(\text{Mugs}) + 0.2\,(\text{Snacks}),
$$

which is hard to point at and name.

In CR, $C$ consists of **actual, unedited columns of $A$**. If 5000 products are reduced to 50 columns in $C$, those are 50 real products that carry all the information in the full set, and $R$ states how the other 4950 are built from them. The price is that CR gives you no ordering by importance and no orthogonality, just an exact factorisation into independent columns.

---

## Code Implementation

`CR_Decomposition.ipynb` runs this idea on a tiny dataset where one feature is deliberately redundant.

### Dataset

Five samples, four features. Three are random integers drawn with `np.random.seed(42)`, and one is built from the others:

| Column | Name | How it is made |
|---|---|---|
| 0 | `Item_Price` | random integers 1–9 |
| 1 | `Shipping_Fee` | random integers 1–9 |
| 2 | `Total_Cost` | `Item_Price + Shipping_Fee` |
| 3 | `Quantity` | random integers 1–9 |

```python
np.random.seed(42)
f0 = np.random.randint(1, 10, size=(5, 1))
f1 = np.random.randint(1, 10, size=(5, 1))
f2 = f0 + f1                       # purely redundant feature
f3 = np.random.randint(1, 10, size=(5, 1))

A = np.hstack([f0, f1, f2, f3]).astype(float)   # shape (5, 4)
```

```
--- Original Dataset Matrix A ---
[[ 7.  3. 10.  8.]
 [ 4.  7. 11.  8.]
 [ 8.  8. 16.  3.]
 [ 5.  5. 10.  6.]
 [ 7.  4. 11.  5.]]
Features: ['Item_Price', 'Shipping_Fee', 'Total_Cost', 'Quantity']
```

Here $a_3 = a_1 + a_2$ (in 1-based numbering), so the rank is 3, not 4.

### Step 1: Row-Reduce to get $R$ and the Pivot Columns

`sympy`'s `rref()` returns both the reduced matrix and the indices of the pivot columns.

```python
sympy_A = Matrix(A)
R_sympy, pivots = sympy_A.rref()
R = np.array(R_sympy).astype(float)
```

### Step 2: Build $C$ from the Pivot Columns of the Original $A$

```python
C = A[:, pivots]
```

```
--- Matrix C (Core Independent Features) ---
[[7. 3. 8.]
 [4. 7. 8.]
 [8. 8. 3.]
 [5. 5. 6.]
 [7. 4. 5.]]
Kept Independent Features: ['Item_Price', 'Shipping_Fee', 'Quantity']

--- Matrix R (The Recipes / Dependencies) ---
[[1. 0. 1. 0.]
 [0. 1. 1. 0.]
 [0. 0. 0. 1.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
Pivot Column Indices: (0, 1, 3)
```

Read the third column of $R$, $(1,1,0)^\top$, with the column-picker view from above: *"one copy of `Item_Price` plus one copy of `Shipping_Fee`"*. That is precisely how `Total_Cost` was built, and the decomposition recovered it without being told.

The other three columns are identity columns, so `Item_Price`, `Shipping_Fee` and `Quantity` are copied through unchanged.

### Step 3: Check that $CR$ gives back $A$?

`rref()` returns an $m\times n$ matrix, which here is $5\times 4$. Only its first `rank` rows are non-zero, and `C` has `rank` columns. So the code trims `R` to those rows before multiplying:

```python
rank = len(pivots)            # 3
R_trimmed = R[:rank, :]       # (3, 4)
reconstruction = C @ R_trimmed   # (5, 3) @ (3, 4) -> (5, 4)
```

```
--- Verification: C @ R reconstructs A ---
[[ 7.  3. 10.  8.]
 [ 4.  7. 11.  8.]
 [ 8.  8. 16.  3.]
 [ 5.  5. 10.  6.]
 [ 7.  4. 11.  5.]]
Matches original A: True
```

### Step 4: Drop the Redundant Columns

Any column that is not a pivot column is a combination of the others and can go.

```python
redundant_indices = list(set(range(A.shape[1])) - set(pivots))
reduced_A = A[:, pivots]
```

```
--- Dimensionality Reduction Summary ---
❌ Deleting feature 'Total_Cost' (Column 2) because it is redundant.

--- Final Reduced Dataset ---
[[7. 3. 8.]
 [4. 7. 8.]
 [8. 8. 3.]
 [5. 5. 6.]
 [7. 4. 5.]]
Remaining Features: ['Item_Price', 'Shipping_Fee', 'Quantity']
```

The reduced dataset is identical to `C`: a $5\times 3$ matrix of three real, named features, with nothing lost, since `Total_Cost` can be recomputed from the other two at any time.

---

## Note:

- **Pivots are chosen left to right.** RREF picks the *first* set of independent columns it meets. Here `Total_Cost` is dropped because it comes after the two columns it depends on. Had it been placed first, it would have been kept and one of its summands dropped, giving an equally valid but different $C$.
- **Exact arithmetic matters.** The code uses `sympy` for the row reduction. The input here is small integers stored as floats, which `sympy` handles cleanly. Real-world data with noisy floating-point values will rarely produce exactly dependent columns, so the rank found this way is the *exact* rank, not a numerical one. For noisy data, SVD-based rank estimation is the more robust tool.
- **At this size there is no storage saving.** $C$ has $5\times3=15$ entries and the non-trivial part of $R$ is tiny, versus $5\times4=20$ entries for $A$. The benefit here is dropping a redundant feature. Storage savings only appear when $r$ is much smaller than $n$.

---

## How to Run the Code

```bash
pip install numpy sympy
jupyter notebook CR_Decomposition.ipynb
```

The notebook also runs directly in Google Colab, the first cell installs `sympy`.
