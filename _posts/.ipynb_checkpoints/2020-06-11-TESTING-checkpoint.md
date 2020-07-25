---
title: TEST MATHJAX
description: "adapt to Lilian's blog"
layout: post
toc: true
categories: [tutorials]
comments: true
---

<!--more-->


## Basic Concepts

[Markov Chain](https://en.wikipedia.org/wiki/Markov_chain)

A Markov process is a ["memoryless"](http://mathworld.wolfram.com/Memoryless.html) (also called "Markov Property") stochastic process. A Markov chain is a type of Markov process containing multiple discrete states. That is being said, the conditional probability of future states of the process is only determined by the current state and does not depend on the past states.


[Kullback–Leibler (KL) Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

KL divergence measures how one probability distribution $$p$$ diverges from a second expected probability distribution $$q$$. It is asymmetric.

$$
\begin{aligned}
D_{KL}(p \| q) &= \sum_x p(x) \log \frac{p(x)}{q(x)} dx \\
 &= - \sum_x p(x)\log q(x) + \sum_x p(x)\log p(x) \\
 &= H(P, Q) - H(P)
\end{aligned}
$$

$$D_{KL}$$ achieves the minimum zero when $$p(x)$$ == $$q(x)$$ everywhere.


[Mutual Information](https://en.wikipedia.org/wiki/Mutual_information)

Mutual information measures the mutual dependence between two variables. It quantifies the "amount of information" obtained about one random variable through the other random variable. Mutual information is symmetric.

$$
\begin{aligned}
I(X;Y) &= D_{KL}[p(x,y) \| p(x)p(y)] \\
 &= \sum_{x \in X, y \in Y} p(x, y) \log(\frac{p(x, y)}{p(x)p(y)}) \\
 &= \sum_{x \in X, y \in Y} p(x, y) \log(\frac{p(x|y)}{p(x)}) \\ 
 &= H(X) - H(X|Y) \\
\end{aligned}
$$














