# Understanding Adam by Watching It Work

This guide is for anyone who has seen the Adam update rule, nodded politely, and still doesn't quite feel what it does. The companion notebook, [`Adam_Optimiser_Comparison.ipynb`](Adam_Optimiser_Comparison.ipynb), implements GD, Momentum, AdaGrad, RMSProp and Adam from scratch and lets you watch them side by side. Here is how to read it, step by step.

## Start with the problem every optimiser faces

Training means walking downhill on a loss surface until you reach a minimum. At each step an optimiser has two decisions to make:

- **Which direction** to move.
- **How far** to move.

The gradient $g$ points uphill, so $-g$ is the obvious direction. The hard part is the step size. A learning rate $\eta$ that is perfect in one direction can be too large or too small in another.

To see this, the notebook uses bowl-shaped losses $L(w)=\tfrac12 w^\top H w$ where the curvature differs a lot between directions:

| Problem | Loss | Valley |
|---|---|---|
| A | $w_1^2+10\,w_2^2$ | mildly stretched, along the axes |
| B | $w_1^2+1000\,w_2^2$ | very stretched, along the axes |
| C | same curvatures as B | very stretched, **rotated 30°** |

![The three test surfaces](images/test_surfaces.png)

Think of B as a long, narrow canyon. Across the canyon the walls are steep (curvature 2000). Along the canyon the floor is almost flat (curvature 2). One learning rate has to serve both.

## Build Adam one idea at a time

Each optimiser below fixes a specific weakness of the one before it. Reading them in this order is the easiest way to understand why Adam looks the way it does.

### Gradient descent: one step size for everything

$$w \leftarrow w-\eta\, g$$

In the canyon, the steep direction decides how large $\eta$ can be. Plain GD is stable only for $\eta<2/\lambda_{\max}$, which is 0.001 on problems B and C. Any larger and it bounces across the walls and blows up. But with such a small $\eta$, progress along the flat floor is painfully slow.

### Momentum: remember where you were going

$$V\leftarrow\beta V+\eta\, g,\qquad w\leftarrow w-V$$

$V$ is a running memory of past gradients. Steps that keep pointing the same way build up speed, and side-to-side bouncing partly cancels out. The cost is overshoot, which you can see as the wiggly orange curve in the first experiment below. Momentum still uses **one** $\eta$ for every direction, so it still has a stability limit tied to the steepest curvature (about $2(1+\beta)/\lambda_{\max}$).

### AdaGrad: give every parameter its own step size

$$G\leftarrow G+g^2,\qquad w\leftarrow w-\eta\,\frac{g}{\sqrt G+\epsilon}$$

Everything here is element-wise, so each parameter divides by its own history. A parameter with large gradients gets small steps; one with small gradients gets large steps. This is a different kind of fix: instead of one global $\eta$, the step is rescaled per coordinate. The catch is that $G$ is a running **sum**, so it only grows and the steps keep shrinking.

### RMSProp: forget old gradients

$$G\leftarrow\beta G+(1-\beta)\,g^2,\qquad w\leftarrow w-\eta\,\frac{g}{\sqrt G+\epsilon}$$

The only change from AdaGrad is replacing the sum with a **moving average**. Old gradients fade, so the effective step size does not die out. (In the notebook, $\beta=0.9$.)

### Adam: Momentum and RMSProp together, with a correction

Adam keeps both kinds of memory:

$$m_t=\beta_1 m_{t-1}+(1-\beta_1)\,g_t \qquad\text{(like Momentum: smoothed direction)}$$

$$v_t=\beta_2 v_{t-1}+(1-\beta_2)\,g_t^{2} \qquad\text{(like RMSProp: smoothed size)}$$

Both start at zero, so early on they are too small. Adam corrects for that:

$$\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}}$$

and then steps

$$w_t=w_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},\qquad \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}.$$

In the code it is only a few lines:

```python
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
```

## The one thing to remember about Adam

Look at the ratio $\hat m_t/\sqrt{\hat v_t}$. If a coordinate's gradient keeps a steady sign and size, then $\hat m_t\approx g$ and $\sqrt{\hat v_t}\approx |g|$, so the ratio is about $\pm1$. The step is therefore about $\eta$ **whatever the size of the gradient**.

Two consequences follow, and the notebook is built to test both:

- On B, the steep and flat directions both get steps of about $\eta$. That is why Adam should cope with bad scaling.
- This rescaling happens **one coordinate at a time**. If the awkward direction is tilted relative to the axes (problem C), there is no coordinate-wise fix. That is why Adam should struggle there.

## How to read the notebook

Run the cells top to bottom. Here is what to look for in each experiment.

### Experiment 1: does Adam win on an easy problem?

Everyone gets the same learning rate ($\eta=0.03$) on problem A.

![Paths and loss curves on problem A](images/exp1_shared_lr.png)

How to read the right-hand plot: the y-axis is a log scale, so a straight downward line means steady, geometric progress. A jagged line means overshooting. A nearly flat line means barely moving.

- Adam finishes at a loss of about $9\times10^{-3}$, well behind GD ($1.7\times10^{-5}$) and RMSProp ($4\times10^{-8}$).
- The jagged orange line is Momentum overshooting.
- The almost flat brown line is AdaGrad: its summed $G$ keeps growing and the steps shrink.
- The printout at the bottom reruns this at $\eta=0.05$, and GD jumps to first place.

What this teaches: Adam is not a "faster" method by default. Its steps are about $\eta$ per coordinate, so crossing a distance of 2 takes roughly $2/\eta\approx70$ steps no matter how gentle the slope. It also teaches that a single learning rate gives a misleading ranking, which is the motivation for the next experiment.

### Experiment 2: how forgiving is each optimiser?

Instead of one $\eta$, this sweeps 25 values from $10^{-4}$ to $1$ and plots the final loss for each. Read it like this: **the width of the dip is how forgiving the optimiser is.** A wide, low valley means you can pick $\eta$ carelessly. A narrow dip means you must tune.

![Final loss versus learning rate](images/exp2_lr_sweep.png)

Learning rates (out of 25) that reached a final loss below $10^{-3}$:

| Optimiser | A | B | C |
|---|---|---|---|
| GD | 5 | 0 | 0 |
| Momentum | 14 | 2 | 2 |
| AdaGrad | 5 | 5 | 0 |
| RMSProp | 1 | 0 | 0 |
| Adam | 11 | 11 | 0 |

Things to spot in the plot:

- **Adam** (red) has a wide low plateau on A and B, from roughly $\eta=0.02$ up to $1$.
- **GD** (purple) and **Momentum** (orange) have curves that simply stop at a certain $\eta$. Those are diverged runs, and the stopping point moves left as the problem gets worse-scaled.
- **AdaGrad** (brown) is excellent on A and B but only at large $\eta$, and its best value is at the edge of the grid, so its true optimum is not found here.
- **RMSProp** (cyan) dips, then rises again as $\eta$ grows: it jitters around the minimum at a scale set by $\eta$.
- In the third panel (rotated valley C), almost nothing converges. Hold that thought.

### Experiment 3a: where Adam shines

Problem B, with $\eta=0.05$. That is 50 times above GD's stability limit.

![Problem B at eta = 0.05](images/exp3a_problem_B.png)

| Optimiser | Final loss |
|---|---|
| GD | diverged |
| Momentum | diverged |
| AdaGrad | 1.995e+00 |
| RMSProp | 6.220e-01 |
| Adam | 6.089e-09 |

GD and Momentum explode, while Adam converges to about $10^{-8}$ with the same $\eta$. This is the per-coordinate rescaling doing its job. (AdaGrad and RMSProp not converging here is not a verdict on them. A shared $\eta$ is just one slice of the sweep, and AdaGrad needs larger $\eta$.)

### Experiment 3b: where Adam stops helping

Problem C is problem B rotated by 30°, with the same curvatures. Each optimiser gets its own best $\eta$ from the sweep, so the comparison is as fair as the grid allows.

![Problem C at each optimiser's best eta](images/exp3b_problem_C.png)

| Optimiser | Best η | Final loss |
|---|---|---|
| GD | 0.0007 | 1.028e+00 |
| Momentum | 0.0015 | 7.760e-06 |
| AdaGrad | 1.0000 | 5.228e-01 |
| RMSProp | 0.0147 | 5.072e-01 |
| Adam | 1.0000 | 9.107e-02 |

Same curvatures as B, yet Adam now ends around $9\times10^{-2}$. Nothing about the *difficulty* changed, only its orientation. Because Adam rescales axis by axis, it cannot undo a tilted valley. Momentum reaches the lowest loss, but look at its curve: it swings wildly over many orders of magnitude, so its final number is partly where an oscillation happened to end. Treat this panel as an illustration of the mechanism, not a benchmark.

Put 3a and 3b together and you get the clearest statement in the notebook: **Adam fixes bad scaling, not bad orientation.**

### Experiment 4: why the bias correction is there

Feed a constant gradient $g=1$. The true averages are exactly 1, but because $m$ and $v$ start at 0, the raw values begin far too small.

![Raw versus bias-corrected moments](images/exp4_bias_correction.png)

```
t = 1 : m = 0.1000  m_hat = 1.0000  |  v = 0.0010  v_hat = 1.0000
t = 30: m = 0.9576  m_hat = 1.0000  |  v = 0.0296  v_hat = 1.0000

First step size WITHOUT bias correction: 0.3162   WITH correction: 0.1000
```

A common guess is that without correction the first step would be too *small*, since $m$ is only 0.1. But $v$ is underestimated far more than $m$ (0.001 versus 0.1), and it sits under a square root:

$$\eta\cdot\frac{0.1}{\sqrt{0.001}}\approx 3.2\,\eta$$

So the uncorrected first step is about **three times too large**. The correction brings it back to exactly $\eta$ and fades away as $t$ grows (about 10 steps for $m$, about 1000 for $v$).

## Try it yourself

The notebook is short, so it is easy to poke at. Before running each change, write down what you expect, then check.

- **Change the learning rate in Experiment 1** (try 0.01, 0.03, 0.05, 0.08). Watch the ranking reshuffle.
- **Turn AdaGrad into RMSProp.** In `adagrad`, change `G = G + g**2` to `G = 0.9 * G + 0.1 * g**2`. That one line is the whole difference between them.
- **Change the angle of problem C.** The notebook only tests 0° and 30°. Try other angles such as 5°, 15° and 45°. How does Adam's best final loss change as the valley tilts?
- **Change Adam's $\beta_2$** from 0.999 to 0.9 and compare with RMSProp. How different do they still look?
- **Remove the bias correction** from `adam` (use `m` and `v` directly) and rerun Experiments 1 and 3a. Is the difference visible in the final loss, or only in the first few steps?
- **Try a different start point** `w0`. All the results above use a single starting point.

## What to take away

- Each optimiser fixes one weakness: Momentum smooths the direction, AdaGrad and RMSProp rescale each parameter, and Adam combines both and corrects the start-up bias.
- Adam's step is about $\eta$ per coordinate, which makes it **forgiving about the learning rate** and good at **badly scaled coordinates**.
- It is **not** the fastest on easy problems, and it does **not** help when the difficulty is tilted relative to the axes.
- Single-learning-rate comparisons mislead. Sweeping $\eta$ tells you much more.

## Limitations

- Only 2-D quadratics, exact gradients, one starting point, a 25-value learning-rate grid and 100 to 200 iterations.
- It says nothing directly about mini-batch training, non-convex losses or neural networks, where Adam is mostly used. What it shows is the *mechanism* behind Adam's behaviour, not a verdict on how it performs in deep learning.
- Several best learning rates (AdaGrad, and Adam on C) sit at the edge of the grid, so their true optima are unknown.

## How to run

```bash
pip install numpy matplotlib jupyter
jupyter notebook Adam_Optimiser_Comparison.ipynb
```

Or upload the notebook to [Google Colab](https://colab.research.google.com/) and run all cells. There is no randomness, so the numbers in this guide should reproduce exactly.
