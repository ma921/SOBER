# SOBER

This repository contains the python code that was presented for the following paper.

[1] Adachi, M., Hayakawa, S., Hamid, S., Jørgensen, M., Oberhauser, H., and Osborne, M. A. SOBER: Highly Parallel Bayesian Optimization and Bayesian Quadrature over Discrete and Mixed Spaces. arXiv 2023 <br>
[arXiv](https://arxiv.org/abs/2301.11832),

![Animate](./docs/animated.gif)

- Red star: ground truth
- black crosses: next batch queries recommended by SOBER
- white dots: historical observations
- Branin function: blackbox function to maximise
- $\pi$: the probability of global optima locations estimated by SOBER

## Brief explanation
![plot](./docs/visual_explanation.png)<br>

We solve batch global optimization as Bayesian quadrature;
$$x^*_\text{true} = $$
\text{argmax}_x f_\text{true}(x)
\delta_{x^*_\text{true}} \in \arg \max_{\pi} \int f_\text{true}(x) \text{d}\pi(x)
\quad \xLeftrightarrow{\text{dual}} \quad 

We query 10 points in parallel to the true blackbox function. Colours represent the GP surrogate model trying to approximate the three true posteriors (Ackley, Oscillatory, Branin-Hoo, see Supplementary Figure 4 for details).
The black dots in the animated GIF is the proposed points by BASQ for each iteration. At the third iteration, BASQ can capture the whole posterior surface.

For understanding SOBER, see tutorials. <br>
For reproducing the results, see examples. <br>
