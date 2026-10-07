# Ill-Conditioned Hessian: Gradient Descent vs Newton's Method

A small, fully worked 2-D example showing **why a single learning rate struggles when the loss surface is much steeper in one direction than another**, and how Newton's method fixes this by using curvature (the Hessian).

The notebook `GDvsNewtons.ipynb` minimises one function with two optimisers from the same starting point and draws both paths on the contour plot.

---

## 1. The idea in one paragraph

Gradient descent (GD) moves by `-η∇L`. The same scalar `η` is applied to every direction, so it has to be small enough for the *steepest* direction to stay stable, and is then painfully slow in the *flat* direction. Newton's method replaces `η` with the inverse Hessian, `H⁻¹`, which divides each direction's step by that direction's curvature: small steps where the surface is steep, large steps where it is flat. On a quadratic function this lands on the minimum in a single step.

## 2. Why it matters in ML

- Loss landscapes of real models are rarely round bowls. Directions with very different curvature are the norm, and the ratio of the largest to smallest Hessian eigenvalue (the **condition number**) measures how bad this is.
- This is the geometric reason behind zig-zagging in plain GD, and the motivation for momentum, adaptive learning rates (AdaGrad / RMSProp / Adam) and quasi-Newton methods. Each of these is a cheaper way of approximating what `H⁻¹` does exactly.
- Newton's method itself needs the full Hessian and its inverse, which is infeasible for large networks. This notebook shows the idea in the smallest setting where it can be checked by hand.

## 3. The function and its derivatives

$$
f(x, y) = 10x^2 + y^2
$$

$$
\nabla f(x, y) = \begin{bmatrix} 20x \\ 2y \end{bmatrix},
\qquad
H = \begin{bmatrix} 20 & 0 \\ 0 & 2 \end{bmatrix},
\qquad
H^{-1} = \begin{bmatrix} 1/20 & 0 \\ 0 & 1/2 \end{bmatrix}
$$

`H` is already diagonal, so its eigenvalues are simply `λ_x = 20` and `λ_y = 2`. The condition number is

$$
\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{20}{2} = 10 .
$$

The level sets `f = c` are ellipses with semi-axes proportional to `1/√λ`, so the ratio of the long axis (along `y`) to the short axis (along `x`) is `√κ ≈ 3.16`. This is the narrow valley seen in the plot: steep across `x`, flat along `y`.

## 4. The two update rules

**Gradient descent**

$$
w_{k+1} = w_k - \eta \nabla f(w_k)
\;\;\Longrightarrow\;\;
x_{k+1} = (1 - 20\eta)\,x_k, \qquad y_{k+1} = (1 - 2\eta)\,y_k
$$

Each coordinate is multiplied by a fixed factor every step. GD converges only if both factors have magnitude below 1, i.e.

$$
|1 - 20\eta| < 1 \;\Rightarrow\; 0 < \eta < \tfrac{2}{\lambda_{\max}} = 0.1 .
$$

**Newton's method**

$$
w_{k+1} = w_k - H^{-1}\nabla f(w_k)
$$

For this function, `H⁻¹∇f(w) = (x, y) = w`, so the update gives `w_{k+1} = 0` regardless of the starting point.

## 5. Code walkthrough

### Objective, gradient and inverse Hessian

```python
def f(x, y):
    return 10 * x**2 + y**2  # Elongated valley function

def grad_f(x, y):
    return np.array([20 * x, 2 * y])

# H = [[20, 0], [0, 2]] -> H_inv = [[1/20, 0], [0, 1/2]]
H_inv = np.array([[1/20, 0.0],
                  [0.0, 1/2]])
```

`H_inv` is written in by hand from the known Hessian; it is not computed numerically.

### Gradient descent: 7 steps from (1.5, 1.5) with η = 0.091

```python
start_pt = np.array([1.5, 1.5])

lr = 0.091
gd_path = [start_pt]
curr_gd = start_pt.copy()
for _ in range(7):
    curr_gd = curr_gd - lr * grad_f(curr_gd[0], curr_gd[1])
    gd_path.append(curr_gd)
gd_path = np.array(gd_path)
```

With `η = 0.091` the two per-coordinate factors are

$$
1 - 20(0.091) = -0.82, \qquad 1 - 2(0.091) = 0.818 .
$$

So `x` flips sign every step and shrinks by 18 % each time, while `y` shrinks smoothly by 18 % and never changes sign.

### Newton's method: one step

```python
newton_path = [start_pt]
newton_step = -H_inv @ grad_f(start_pt[0], start_pt[1])
newton_path.append(start_pt + newton_step)
newton_path = np.array(newton_path)
```

### Plot

The remaining code builds a 400 x 400 grid, draws the contours of `f` at levels `[0.2, 1, 3, 7, 13, 22, 35]`, overlays the red GD path and the blue dashed Newton path, and marks the start point and the minimum.

## 6. Results




- **Newton (blue, dashed):** a single straight segment from `(1.5, 1.5)` to `(0, 0)`. The path points directly at the minimum because `H⁻¹` rescales the elliptical contours into circles, in which the gradient points at the centre.
- **Gradient descent (red):** a zig-zag. Each step jumps across the valley in `x` (the steep direction) while creeping towards 0 in `y`. After 7 steps it is still at about `(-0.37, 0.37)`, not at the minimum.

**Numbers behind the red path** (replaying the same update rule as the notebook; these are not printed by the notebook itself):

| Step | x | y | f(x, y) |
|---:|---:|---:|---:|
| 0 | 1.5000 | 1.5000 | 24.750 |
| 1 | -1.2300 | 1.2270 | 16.635 |
| 2 | 1.0086 | 1.0037 | 11.180 |
| 3 | -0.8271 | 0.8210 | 7.514 |
| 4 | 0.6782 | 0.6716 | 5.050 |
| 5 | -0.5561 | 0.5494 | 3.394 |
| 6 | 0.4560 | 0.4494 | 2.281 |
| 7 | -0.3739 | 0.3676 | 1.533 |
| Newton, step 1 | 0 | 0 | 0 |

**A point worth noticing.** `η = 0.091` is not an arbitrary "bad" step size. For a quadratic with eigenvalues `λ_min` and `λ_max`, the fixed step that minimises the worst per-coordinate factor is

$$
\eta^\* = \frac{2}{\lambda_{\max} + \lambda_{\min}} = \frac{2}{22} \approx 0.0909,
\qquad
\text{rate} = \frac{\kappa - 1}{\kappa + 1} = \frac{9}{11} \approx 0.818 .
$$

So the GD run shown is essentially the *best* a single fixed learning rate can do here. The slowness is not a tuning mistake: it is caused by `κ = 10`. Making `η` larger would push `|1 - 20η|` above 1 and GD would diverge along `x`; making it smaller would slow `y` down further.

## 7. What this example does and does not show

- It shows the effect of conditioning on GD, and Newton's one-step convergence on a **quadratic**. The one-step result is exact only because `f` is exactly quadratic; on a general loss, Newton's method is iterative.
- `H` is positive definite here (both eigenvalues > 0), so Newton moves towards a minimum. At a saddle or maximum the Hessian has negative eigenvalues and the plain Newton step does not behave this way.
- Newton's method is not compared on cost. Here `H` is 2 x 2; for a network with `d` parameters it is `d x d`, and forming and inverting it costs O(d³).

## 8. How to run

```bash
pip install numpy matplotlib jupyter
jupyter notebook illhessianNewtonandGD.ipynb
```

Run the single cell. It uses a fixed start point and no randomness, so the figure is reproducible.
