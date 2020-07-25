---
title: Vector Markdown
description: "check compatibility"
layout: post
toc: true
categories: [tutorials]
author: Aditya Rana, Amey Agrawal and Nikhil Verma
comments: true
---

<!--more-->

### Situation

In a hypothetical $n$-dimensional universe, there exists $p$ population of a particular species of human, Homo BITSians. These species like to hangout in specialized eateries, called Redi. In this universe, there are $q$ Redis which serve delicious snacks and beverages at nominal prices. Our task is to find the nearest Redi from each of the Homo BITSians so that they spend less time on commuting. Another part of the problem is to give the number of Redis inside a radius of $r$ metres from each Homo BITSians which will help them to hangout in as many Redis as possible.
### Problem
Matrices, $X \in p \times n$ and $Y \in q \times n$, which has the co-ordinates of $p$ Homo BITSians and $q$ Redis respectively in the $n$-dimensional universe are given. The $i^{th}$ row in the matrix, $X$, corresponds to the $i^{th}$ Homo BITSian. Similarly, the $i^{th}$ row in the matrix, $Y$, corresponds to the $i^{th}$ Redi.

**Note**: Here, row numbering (indexing) starts from $0$.
### Task

Given $X$, $Y$, find a vector, $V$, of length $p$. The vector, $V$, is such that the $i^{th}$ element of $V$ has the index of the nearest Redi from the $i^{th}$ Homo BITSian.

Distance metric is the usual $l_2$-norm.
In a n-dimensional space with points $x = (x_0, x_0, \ldots, x_{n-1})$ and $y = (y_0, y_0, \ldots, y_{n-1})$, the distance can be calculated as:

$$D_{xy}^2 = (x_0 - y_0)^2 
+ (x_1 - y_1)^2 + \ldots + (x_{n-1} - y_{n-1})^2$$

### Part 1: Find the index of the nearest Redi from each Homo BITSian

````python

# Base Distance Function to be completed by the student

import numpy as np

def distances(X, Y):
    """
    Given matrices X and Y, the function returns a distance matrix. 
    The (i,j)th element of the matrix contains the distance of jth Redi 
    from the ith Homo BITSian.
    
    Parameters: X,Y
    Returns: D
    """
    
    ### BEGIN SOLUTION

    ### END SOLUTION
    
    return D
````

### Solutions begin from here

The way to understand the **axis** argument in numpy functions is that it collapses the specified axis. So when we specify the axis 1 (the column), it applies the function across all columns, resulting in a single column.For more intuition check out this [post](https://medium.com/@aerinykim/numpy-sum-axis-intuition-6eb94926a5d1) by Aerin Kim. In fact, I would reccommend you to read all of her posts.

````python
def nearest_redi(X, Y):
    D = distances(X, Y)
    V = np.argmin(D, 1)
    return V
````











