# Change of Basis

> **Core idea:** a vector is a point in space. Its coordinates are only a description of that point, and the description depends on which basis you use to look at it. Changing the basis changes the numbers, never the point.

This module uses a small 2D example, with a plot at each step, to build that idea. The notebook (`COB.ipynb`) contains the step-by-step derivation. This README explains the concept behind it and what the results show.

---

## The concept

### A vector is not its coordinates

Take the pair of numbers $(3, 3)$. It is tempting to treat this pair as the vector itself. It is not. The pair is a set of instructions: *"take 3 steps along the first reference direction, then 3 steps along the second."* The arrow in space is the actual object. The numbers only say how to build it from a chosen pair of reference directions.

By default we use the **standard basis** $\mathbf{e}_1=(1,0)$ and $\mathbf{e}_2=(0,1)$, which are "right" and "up". This is a convention, and nothing forces it. Any two linearly independent vectors can serve as reference directions, and each choice gives every vector in the plane a different numerical description.

### Changing basis means changing the lens

Imagine two observers looking at the same arrow:

| | Observer 1 | Observer 2 |
|---|---|---|
| Reference directions | $\mathbf{e}_1=(1,0),\ \mathbf{e}_2=(0,1)$ | $\mathbf{b}_1=(2,1),\ \mathbf{b}_2=(-1,1)$ |
| Their grid | square | skewed (parallelogram cells) |
| Description of the arrow | $3\,\mathbf{e}_1 + 3\,\mathbf{e}_2$ | $2\,\mathbf{b}_1 + 1\,\mathbf{b}_2$ |
| Coordinates | $(3,\ 3)$ | $(2,\ 1)$ |

They write different numbers, but they are pointing at the same arrow. A change of basis is the translation between these two descriptions. Nothing happens to the vector.

The matrix $P$, whose columns are the new basis vectors written in standard coordinates, performs this translation:

$$
\underbrace{P\,[\mathbf{v}]_B}_{\text{B-lens} \to \text{standard lens}} = \mathbf{v}, \qquad 
\underbrace{P^{-1}\mathbf{v}}_{\text{standard lens} \to \text{B-lens}} = [\mathbf{v}]_B
$$

Geometrically, $P$ is the linear map that carries the square grid onto the observer-2 grid ($\mathbf{e}_1\mapsto\mathbf{b}_1$, $\mathbf{e}_2\mapsto\mathbf{b}_2$). That is why the new basis vectors are its columns.

### Why this matters in machine learning

Almost every vector in ML (a data row, an embedding, an image) is a coordinate vector relative to some implicit basis, usually "one axis per feature". Many methods amount to finding a better basis for the same data:

- **PCA** re-expresses data in a basis aligned with the directions of maximum variance, so a few coordinates carry most of the information.
- **Diagonalisation and SVD** pick the basis in which a matrix acts as pure scaling, which makes it easy to analyse.
- **Hidden layers** in a network can be read as re-describing the data in a new, learned coordinate system at each layer.

Understanding that the data stays the same while its description changes is the foundation for all of these.

---

## Code Implementation

The notebook runs one concrete example, with $\mathbf{v}=(3,3)$ in the standard basis and the new basis $\mathbf{b}_1=(2,1)$, $\mathbf{b}_2=(-1,1)$. The core of the computation is only a few lines:

```python
P = np.column_stack((b1, b2))   # new basis vectors as columns
P_inv = np.linalg.inv(P)

v_new = P_inv @ v               # standard -> new basis coordinates
recovered = P @ v_new           # new basis -> standard (round trip)
```

Two helper functions then draw the new basis grid. `get_grid_ranges` converts the plot corners into B-coordinates to decide how many grid lines are needed. `draw_new_basis_grid` draws the lines $i\,\mathbf{b}_1 + j\,\mathbf{b}_2$ for integer $i, j$, which is the same construction as ordinary grid lines but along $\mathbf{b}_1$ and $\mathbf{b}_2$.

---

## Results

### Numerical output

| Quantity | Value |
|---|---|
| $\det P$ | $3$ (non-zero, so the vectors form a valid basis) |
| $P^{-1}$ | $\tfrac13\begin{bmatrix}1&1\\-1&2\end{bmatrix}$ |
| $[\mathbf{v}]_B = P^{-1}\mathbf{v}$ | $(2,\ 1)$ |
| $P\,[\mathbf{v}]_B$ | $(3,\ 3)$, which is the original vector |
| Recovered $-$ original | $(0,\ 0)$ |

The round trip returns the original vector exactly, so the two descriptions are consistent.

### Plot 1: two grids, one vector

![Standard grid (grey, solid) and new basis grid (teal, dashed) with the same vector](images/grids_overlay.png)

The grey grid is the standard lens and the teal dashed grid is the new one. The red arrow is a single object that sits on both grids. Read against the grey grid, it ends at $(3,3)$. Read against the teal grid, it ends at the lattice point reached by 2 steps of $\mathbf{b}_1$ and 1 step of $\mathbf{b}_2$. The teal cells are parallelograms, and each has area $|\det P| = 3$ times that of a unit square.

### Plot 2: two recipes for the same arrow

![The vector built as 2 b1 + 1 b2 and as 3 e1 + 3 e2](images/two_recipes.png)

The plot shows both ways of building the red vector, each path ending at its tip:

- **New-basis recipe:** go $2\mathbf{b}_1$ (orange), then $1\mathbf{b}_2$ (purple).
- **Standard recipe:** go $3\mathbf{e}_1$ (grey), then $3\mathbf{e}_2$ (black).

The two recipes use different steps and different numbers, but both end at the same tip. This is the whole idea in one picture: the coordinates are instructions, and the arrow is what the instructions build.

---

## How to run

```bash
pip install numpy matplotlib jupyter
jupyter notebook ChangeOfBasis.ipynb
```

The notebook also runs unchanged in Google Colab. Run the cells in order, because the plotting cells reuse variables defined earlier.

