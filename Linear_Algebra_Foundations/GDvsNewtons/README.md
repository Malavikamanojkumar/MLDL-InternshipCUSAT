# Ill-Conditioned Hessian: Gradient Descent vs Newton's Method

In this module we shall see why gradient descent struggles on an ill-conditioned loss surface, and how Newton's method fixes it by using the Hessian (curvature).

\---

## Theory

Gradient descent (GD) moves by `-η∇L`. The same scalar `η` is applied to every direction, so it has to be small enough for the *steepest* direction to stay stable, and is then painfully slow in the *flat* direction. Newton's method replaces `η` with the inverse Hessian, `H⁻¹`, which divides each direction's step by that direction's curvature: small steps where the surface is steep, large steps where it is flat. On a quadratic function this lands on the minimum in a single step.

## Role in Machine Learning

* Loss landscapes of real models are rarely round bowls. Directions with very different curvature are the norm, and the ratio of the largest to smallest Hessian eigenvalue (the **condition number**) measures how bad this is.
* This is the geometric reason behind zig-zagging in plain GD, and the motivation for momentum, adaptive learning rates (AdaGrad / RMSProp / Adam) and quasi-Newton methods. Each of these is a cheaper way of approximating what `H⁻¹` does exactly.
* Newton's method itself needs the full Hessian and its inverse, which is infeasible for large networks. This notebook shows the idea in the smallest setting where it can be checked by hand.

## Setup

We shall use a simple function, f(x, y) = 10x² + y². Think of it as a loss surface shaped like a long, narrow valley. It is very steep in the x direction (curvature 20) and very flat in the y direction (curvature 2). When one direction is much steeper than another, the surface is called ill-conditioned. Here the ratio is 20 : 2 = 10.



$$
f(x, y) = 10x^2 + y^2
$$

$$
\\nabla f(x, y) = \\begin{bmatrix} 20x \\ 2y \\end{bmatrix},
\\qquad
H = \\begin{bmatrix} 20 \& 0 \\ 0 \& 2 \\end{bmatrix},
\\qquad
H^{-1} = \\begin{bmatrix} 1/20 \& 0 \\ 0 \& 1/2 \\end{bmatrix}
$$

`H` is already diagonal, so its eigenvalues are simply `λ\_x = 20` and `λ\_y = 2`. The condition number is

$$
\\kappa(H) = \\frac{\\lambda\_{\\max}}{\\lambda\_{\\min}} = \\frac{20}{2} = 10 .
$$

The level sets `f = c` are ellipses with semi-axes proportional to `1/√λ`, so the ratio of the long axis (along `y`) to the short axis (along `x`) is `√κ ≈ 3.16`. This is the narrow valley seen in the plot: steep across `x`, flat along `y`.

## The two update rules

**Gradient descent**

$$
w\_{k+1} = w\_k - \\eta \\nabla f(w\_k)
;;\\Longrightarrow;;
x\_{k+1} = (1 - 20\\eta),x\_k, \\qquad y\_{k+1} = (1 - 2\\eta),y\_k
$$

Each coordinate is multiplied by a fixed factor every step. GD converges only if both factors have magnitude below 1, i.e.

$$
|1 - 20\\eta| < 1 ;\\Rightarrow; 0 < \\eta < \\tfrac{2}{\\lambda\_{\\max}} = 0.1 .
$$

**Newton's method**

$$
w\_{k+1} = w\_k - H^{-1}\\nabla f(w\_k)
$$

For this function, `H⁻¹∇f(w) = (x, y) = w`, so the update gives `w\_{k+1} = 0` regardless of the starting point.

## Code Implementation

The notebook `GDvsNewtons.ipynb` minimises one function with two optimisers from the same starting point and draws both paths on the contour plot.



```python
def f(x, y):
    return 10 \* x\*\*2 + y\*\*2  # Elongated valley function

def grad\_f(x, y):
    return np.array(\[20 \* x, 2 \* y])

# H = \[\[20, 0], \[0, 2]] -> H\_inv = \[\[1/20, 0], \[0, 1/2]]
H\_inv = np.array(\[\[1/20, 0.0],
                  \[0.0, 1/2]])
```

`H\_inv` is written in by hand from the known Hessian; it is not computed numerically.

### Gradient descent:

```python
start\_pt = np.array(\[1.5, 1.5])

lr = 0.091
gd\_path = \[start\_pt]
curr\_gd = start\_pt.copy()
for \_ in range(7):
    curr\_gd = curr\_gd - lr \* grad\_f(curr\_gd\[0], curr\_gd\[1])
    gd\_path.append(curr\_gd)
gd\_path = np.array(gd\_path)
```

With `η = 0.091` the two per-coordinate factors are

$$
1 - 20(0.091) = -0.82, \\qquad 1 - 2(0.091) = 0.818 .
$$

So `x` flips sign every step and shrinks by 18 % each time, while `y` shrinks smoothly by 18 % and never changes sign.



The code runs 7 steps of w ← w − η∇f from (1.5, 1.5). The learning rate η is a single number(η=0.091) applied to every direction, which causes two problems:



In the steep x direction, the step overshoots the valley floor. x flips sign every step, which produces the zig-zag in the plot.

In the flat y direction, the same step is tiny, so progress toward the minimum is slow.



If you make η bigger to speed up y, x overshoots more and eventually diverges (this happens above η = 0.1). If you make it smaller to stop the zig-zag, y crawls even slower. One learning rate can't suit both directions.

### Newton's method

```python
newton\_path = \[start\_pt]
newton\_step = -H\_inv @ grad\_f(start\_pt\[0], start\_pt\[1])
newton\_path.append(start\_pt + newton\_step)
newton\_path = np.array(newton\_path)
```



The code takes one step of w ← w − H⁻¹∇f, where H is the Hessian matrix of second derivatives. Multiplying by H⁻¹ divides each direction's step by that direction's curvature. The steep x direction gets a small step (1/20), and the flat y direction gets a big step (1/2). Mathematically, this reshapes the narrow elliptical valley into a round bowl, where the gradient points straight at the minimum. For a quadratic function like this one, it reaches (0, 0) in a single step.

### Plot

The remaining code builds a 400 x 400 grid, draws the contours of `f` at levels `\[0.2, 1, 3, 7, 13, 22, 35]`, overlays the red GD path and the blue dashed Newton path, and marks the start point and the minimum.



It draws the contour lines of the valley, the red zig-zag path of gradient descent still far from the minimum after 7 steps, and the blue straight line of Newton's method going directly to the minimum.



## Results





* **Newton (blue, dashed):** a single straight segment from `(1.5, 1.5)` to `(0, 0)`. The path points directly at the minimum because `H⁻¹` rescales the elliptical contours into circles, in which the gradient points at the centre.
* **Gradient descent (red):** a zig-zag. Each step jumps across the valley in `x` (the steep direction) while creeping towards 0 in `y`. After 7 steps it is still at about `(-0.37, 0.37)`, not at the minimum.

**Numbers behind the red path** (replaying the same update rule as the notebook; these are not printed by the notebook itself):

|Step|x|y|f(x, y)|
|-:|-:|-:|-:|
|0|1.5000|1.5000|24.750|
|1|-1.2300|1.2270|16.635|
|2|1.0086|1.0037|11.180|
|3|-0.8271|0.8210|7.514|
|4|0.6782|0.6716|5.050|
|5|-0.5561|0.5494|3.394|
|6|0.4560|0.4494|2.281|
|7|-0.3739|0.3676|1.533|
|Newton, step 1|0|0|0|

**NOTE:** `η = 0.091` is not an arbitrary "bad" step size. For a quadratic with eigenvalues `λ\_min` and `λ\_max`, the fixed step that minimises the worst per-coordinate factor is

$$
\\eta^\* = \\frac{2}{\\lambda\_{\\max} + \\lambda\_{\\min}} = \\frac{2}{22} \\approx 0.0909,
\\qquad
\\text{rate} = \\frac{\\kappa - 1}{\\kappa + 1} = \\frac{9}{11} \\approx 0.818 .
$$

So the GD run shown is essentially the *best* a single fixed learning rate can do here. The slowness is not a tuning mistake: it is caused by `κ = 10`. Making `η` larger would push `|1 - 20η|` above 1 and GD would diverge along `x`; making it smaller would slow `y` down further.

## 

