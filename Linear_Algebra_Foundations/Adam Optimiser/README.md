# Adam vs. GD, Momentum, AdaGrad and RMSProp

Adam is usually described as "Momentum plus RMSProp" and then used without further thought. This module implements all five optimisers from scratch in NumPy and runs them on three small quadratic loss surfaces whose geometry is known exactly, to check what that description actually buys you, and when it doesn't.

The update rule, for reference:

$$m\_t=\\beta\_1 m\_{t-1}+(1-\\beta\_1)g\_t,\\quad v\_t=\\beta\_2 v\_{t-1}+(1-\\beta\_2)g\_t^2,\\quad w\_t=w\_{t-1}-\\eta,\\frac{\\hat m\_t}{\\sqrt{\\hat v\_t}+\\epsilon}$$

with $\\hat m\_t=m\_t/(1-\\beta\_1^t)$ and $\\hat v\_t=v\_t/(1-\\beta\_2^t)$. Dividing by $\\sqrt{\\hat v\_t}$ makes each parameter's step about $\\eta$ regardless of gradient size, which makes Adam a per-coordinate (diagonal) rescaling. Most of what follows is a consequence of that.

## Experiments

||Loss|Condition number|Valley|
|-|-|-|-|
|**A**|$w\_1^2+10w\_2^2$|10|aligned with the axes|
|**B**|$w\_1^2+1000w\_2^2$|1000|aligned with the axes|
|**C**|same curvatures as B|1000|rotated 30°|

## Results

Each optimiser was run for 200 iterations at 25 learning rates between $10^{-4}$ and 1. The table counts how many of those 25 rates reach a final loss below $10^{-3}$ (the starting loss is about 10).

|Learning rates that converge (of 25)|A|B|C|
|-|-|-|-|
|GD|5|0|0|
|Momentum|14|2|2|
|AdaGrad|5|5|0|
|RMSProp|1|0|0|
|**Adam**|**11**|**11**|0|

!\[Final loss versus learning rate on A, B and C](figures/03\_lr\_sweep.png)

* **Adam is forgiving, not fastest.** On A, with one shared learning rate, it finishes behind RMSProp and GD, because its steps are about $\\eta$ per coordinate and cannot be larger on an easy slope. What it offers is a wide working range: roughly $\\eta=0.02$ to 1 on both A and B. On B, at $\\eta=0.05$ (fifty times GD's stability limit), GD and Momentum diverge while Adam reaches a loss of about $10^{-8}$.
* **Rankings at a single learning rate are fragile.** The order on A changes when $\\eta$ moves from 0.03 to 0.05, so the sweep says more than any one run.
* **Adam's rescaling is per-coordinate, so a rotated valley defeats it.** On C no adaptive method converges in 200 iterations, and Adam's best loss is about 0.09. Momentum, tuned to a narrow window of learning rates, reaches about $10^{-5}$, though its loss oscillates violently on the way.

!\[Problem C, each optimiser at its best learning rate](figures/05\_exp3b\_problem\_C.png)

* **AdaGrad does very well here at large $\\eta$** (about 0.2 and up), since on a deterministic convex problem its shrinking step is harmless. At the small learning rates used elsewhere it barely moves.
* **Bias correction removes a first step that is about 3× too large**, not too small: the uncorrected step is $\\eta\\cdot0.1/\\sqrt{0.001}\\approx3.2\\eta$ because $v$ is underestimated far more than $m$.



