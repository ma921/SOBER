# SOBER

This repository contains the python code that was presented for the following paper.

[1] Adachi, M., Hayakawa, S., Hamid, S., Jørgensen, M., Oberhauser, H., and Osborne, M. A. SOBER: Highly Parallel Bayesian Optimization and Bayesian Quadrature over Discrete and Mixed Spaces. arXiv 2023 <br>
[arXiv](https://arxiv.org/abs/2301.11832),

![Animate](./docs/animated.gif)

## Brief explanation
![plot](./docs/visual_explanation.png)<br>

We query 100 points in parallel to the true posterior distribution. Colours represent the GP surrogate model trying to approximate the three true posteriors (Ackley, Oscillatory, Branin-Hoo, see Supplementary Figure 4 for details).
The black dots in the animated GIF is the proposed points by BASQ for each iteration. At the third iteration, BASQ can capture the whole posterior surface.

For understanding SOBER, see tutorials. <br>
For reproducing the results, see examples. <br>
