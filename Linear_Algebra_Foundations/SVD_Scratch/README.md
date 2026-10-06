# Singular Value Decomposition (SVD)

Compressing an image with SVD, using only NumPy, with the decomposition built by hand from an eigendecomposition.

## Definition

Any real matrix $A \in \mathbb{R}^{m \times n}$ can be factored as

$$A = U \Sigma V^T$$

| Symbol | What it is | Geometric role |
|---|---|---|
| $V^T$ | orthogonal matrix, rows are the right singular vectors | rotate / reflect the input |
| $\Sigma$ | diagonal, entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ (singular values) | stretch along each axis |
| $U$ | orthogonal matrix, columns are the left singular vectors | rotate / reflect into the output |

Every linear map is therefore "rotate, stretch, rotate".

## Application of SVD in Machine Learning:

- **Compression / dimensionality reduction:** keep only the largest singular values and drop the rest (this module).
- **Low-rank structure:** real data (images, user–item tables, embeddings) is often close to low rank, so a few directions carry most of the information.
- **Rank and conditioning:** the number of non-zero $\sigma_i$ is the rank of $A$, and $\sigma_1/\sigma_r$ measures how ill-conditioned it is.
- **PCA** is SVD applied to mean-centred data.

## Derivation

SVD is computed here by reducing it to an eigenvalue problem on a symmetric matrix.

1. Substituting $A = U\Sigma V^T$ gives $A^TA = V \Sigma^T\Sigma V^T$. So the columns of $V$ are eigenvectors of $A^TA$:

$$A^T A\, v_i = \lambda_i v_i, \qquad \sigma_i = \sqrt{\lambda_i}$$

2. $A^TA$ is symmetric positive semi-definite, since $x^TA^TAx = \|Ax\|^2 \ge 0$. Its eigenvalues are therefore real and non-negative, and its eigenvectors can be chosen orthonormal. This is why $\sigma_i = \sqrt{\lambda_i}$ always exists and $V$ is orthogonal.

3. From $Av_i = \sigma_i u_i$ we recover the left singular vectors:

$$u_i = \frac{1}{\sigma_i} A v_i \qquad (\sigma_i \neq 0)$$

**Low-rank approximation.** $A$ is a sum of rank-1 pieces, ordered by importance:

$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T \quad\Longrightarrow\quad A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

$A_k$ is the best rank-$k$ approximation of $A$ in the least-squares sense (Eckart–Young theorem).

**Storage.** Storing $A$ costs $mn$ numbers. Storing $A_k$ costs $mk + k + kn$, so

$$\text{compression ratio} = \frac{mn}{mk + k + kn}$$

## Implementation

### Step 1: Load the image as a matrix

The RGB image is converted to grayscale by averaging the three channels, then scaled to $[0,1]$. The resulting matrix is $A$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

china = load_sample_image("china.jpg")

# Grayscale: average across the RGB channels, normalise to [0, 1]
A = np.mean(china, axis=2) / 255.0

plt.figure(figsize=(8, 6))
plt.imshow(A, cmap='gray')
plt.title(f"Original Grayscale Image (Matrix A: {A.shape[0]}x{A.shape[1]})")
plt.axis('off')
plt.show()
```

The sample image is $427 \times 640$ pixels, so $A \in \mathbb{R}^{427 \times 640}$ (here $m = 427 < n = 640$).

### Step 2: SVD from scratch

```python
def svd_from_scratch(A):
    m, n = A.shape

    # 1. A^T A: n x n symmetric matrix; its eigenvectors are V
    ATA = A.T @ A

    # 2. Eigendecomposition (eigh is for symmetric matrices; returns ascending order)
    lambdas, V = np.linalg.eigh(ATA)

    # 3. Sort descending so that sigma_1 >= sigma_2 >= ...
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    V = V[:, idx]

    # 4. sigma = sqrt(lambda); clip tiny negatives caused by floating-point error
    sigmas = np.sqrt(np.maximum(lambdas, 0))

    # 5. Left singular vectors: u_i = A v_i / sigma_i  (only where sigma_i is non-zero)
    U = np.zeros((m, len(sigmas)))
    for i in range(len(sigmas)):
        if sigmas[i] > 1e-15:
            U[:, i] = (A @ V[:, i]) / sigmas[i]

    return U, sigmas, V.T

U, sigmas, Vt = svd_from_scratch(A)
print(f"U shape: {U.shape}, Sigma count: {len(sigmas)}, Vt shape: {Vt.shape}")
```

**Output**

```
U shape: (427, 640), Sigma count: 640, Vt shape: (640, 640)
```

Because the code eigendecomposes the $n \times n$ matrix $A^TA$, it returns $n = 640$ singular values. Since $\text{rank}(A) \le \min(m,n) = 427$, only the first 427 are genuine. The remaining 213 are numerical zeros (about $10^{-6}$ or exactly 0), and the matching columns of `U` are zero or unreliable. This does not affect the reconstructions below, which only use the largest singular values.

### Step 3: Check against NumPy

```python
r = min(A.shape)
U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Max difference in singular values:", np.max(np.abs(sigmas[:r] - s_np)))
print("Full reconstruction error:", np.max(np.abs(U @ np.diag(sigmas) @ Vt - A)))
```

**Output**

```
Max difference in singular values: 7.56e-12
Full reconstruction error: 1.46e-10
```

The hand-built singular values match `np.linalg.svd` to about $10^{-11}$, and multiplying the three factors back together recovers $A$.

### Step 4: Rank-$k$ compression

```python
def compress_image(U, sigmas, Vt, k):
    U_k = U[:, :k]
    S_k = np.diag(sigmas[:k])
    Vt_k = Vt[:k, :]
    return U_k @ S_k @ Vt_k

ks = [5, 20, 50, 100]
plt.figure(figsize=(20, 10))

m, n = A.shape
original_size = m * n

for i, k in enumerate(ks):
    A_k = compress_image(U, sigmas, Vt, k)

    compressed_size = (m * k) + k + (k * n)
    ratio = original_size / compressed_size
    rel_err = np.linalg.norm(A - A_k) / np.linalg.norm(A)

    plt.subplot(1, len(ks), i + 1)
    plt.imshow(A_k, cmap='gray')
    plt.title(f"k = {k}\nRatio: {ratio:.2f}:1\nError: {rel_err:.1%}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## Results

<img width="1696" height="347" alt="svd_reconstructions" src="https://github.com/user-attachments/assets/9cc82a82-52fe-46ac-b174-072476e31afa" />

| k | Compression ratio | Relative error $\|A-A_k\|_F/\|A\|_F$ |
|---|---|---|
| 5 | 51.18 : 1 | 18.4 % |
| 20 | 12.79 : 1 | 13.6 % |
| 50 | 5.12 : 1 | 10.3 % |
| 100 | 2.56 : 1 | 7.4 % |

**Reading the images**

- **k = 5:** only the coarse layout survives. The pagoda shows up as a blurred dark block with horizontal and vertical streaks, which is what a sum of five rank-1 matrices looks like.
- **k = 20:** the pagoda, its roofs and the shoreline are clearly recognisable. Fine texture (trees, water) is still smooth.
- **k = 50 and k = 100:** hard to tell from the original at a glance. Fine detail in the foliage returns gradually.

**Why so few values go so far?** The first singular value is about 327, the second about 60, and by $k=100$ it is about 2.9. The spectrum drops quickly. Using $\sigma_i^2$ as the "energy" of each component, the top 5 values hold about 96.6 % of the total, the top 20 about 98.1 %, and the top 100 about 99.5 %. A natural image is close to low rank, so a handful of directions carries most of what the eye sees.

**The trade-off:** More components give a better image but a lower compression ratio. At $k = 100$ the image is about 7 % off but only 2.6× smaller, which is why the notebook compares several values of $k$ rather than picking one.

## How to run

```bash
pip install numpy matplotlib scikit-learn
jupyter notebook SVD_Scratch.ipynb
```

`load_sample_image("china.jpg")` ships with scikit-learn, so no download is needed. The notebook also runs directly in Google Colab.

## Limitations

- Forming $A^TA$ squares the condition number, so very small singular values are inaccurate (this is why the last 213 values here are noise). It is fine for illustrating the idea, but `np.linalg.svd` uses more stable algorithms.
- `U` is only meaningful for columns with non-negligible $\sigma_i$, so the code is intended for rank-$k$ truncation with small $k$.
- The compression ratio counts stored numbers only. It is not a file-size comparison and ignores quantisation or entropy coding.
