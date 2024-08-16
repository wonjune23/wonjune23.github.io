---
title: Nonlinear Optimization Methods
date: 2024-08-15 20:52:00 +0900
categories: [theory, mathematics]
tags: [gradient descent, gauss newton, levenberg marquardt, lm, nonlinear, least squares, optimization]     # TAG names should always be lowercase
description: A review of nonlinear least squares optimization methods used in computer vision problems.
math: true
---

### Introduction

I mean, you can simply use `scipy.optimize`, `torch`, or `ceres-solver`, to solve a nonlinear optimization problem. So, why bother studying these?
It's because as a *professional* computer vision engineer, you sometimes need to tackle the core of these optimization techniques: maybe your team implemented their own optimization library, and you need to handle errors in it (true story). To do so, the underlying, fundamental knowledge about this topic is required.

Now, there are basically 3 optimization techniques that we commonly encounter in this field: gradient descent (and its variants), Gauss-Newton method, and Levenberg-Marquardt (LM) method. I've seen someone using `BFGS`, but I'd argue that it's really a rare case in this field.

Okay, cool, but, what really is the `gradient` in the first place? 
What is the concrete, mathematical definition of `Jacobian` or `Hessian` that we've heard while learning these methods?

### Definitions

#### Gradient
Gradient is defined over a **scalar-valued** differentiable function $$f$$ that depends on $$n$$ variables. The gradient $$∇f$$ is then a vector field (vector-valued function), that represents the direction and the rate of steepest slope of $$f$$ at a given point. You know, it's just an array of partial derivatives of $$f$$.

$$∇f_{i} = \frac{\partial {f}}{\partial{x_i}},$$

$$∇f: \mathbb{R}^n \to \mathbb{R}^n, f: \mathbb{R}^n \to \mathbb{R}.$$


#### Jacobian
Jacobian on the other hand, is defined over a **vector-valued** differentiable function $$\boldsymbol{f}$$ (notice the notation is now bold). As was in gradient, it's just an array of all its partial derivatives, but this time it forms a matrix, because the output domain is not a scalar but a vector. Say that the $$\boldsymbol{f}$$ depends on $$n$$ variables, and it maps those variables to an $$m$$ dimensional vector. Then, the Jacobian $$J$$ of $$\boldsymbol{f}$$ is an $$m\times n$$ matrix. 
In other words, it's just a list of gradients of length $$m$$, where each gradient corresponds to each scalar element of the $$m$$ dimensional output vector.


$$J_{i,j} = \frac{\partial \boldsymbol{f}_i}{\partial{x_j}},$$

$$J \in \mathbb{R}^{m \times n}, \boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m.$$

#### Hessian
Hessian is again defined over a **scalar-valued** function $$f$$ depending on $$n$$ variables. The Hessian matrix $$H$$ of $$f$$ is a square matrix of shape $$n\times n$$ and it represents all the second-order partial derivatives of $$f$$, describing the local curvature of $$f$$. FYI, if all the second partial derivatives are continuous, then the $$H$$ is symmetric (well, so mathematicians say).


$$H_{i,j} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},$$

$$H \in \mathbb{R}^{n \times n}, f: \mathbb{R}^n \to \mathbb{R}.$$

#### Taylor Expansion of a Multivariable Function

The second-order Taylor expansion of a multivariable scalar function $$f$$, can be written compactly using Jacobian and Hessian of $$f$$. That is:

$$y = f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^{\mathrm{T}} \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^{\mathrm{T}} \mathbf{H}(\mathbf{x}) \Delta \mathbf{x}.$$

I wish I could prove all the details of the mathematics behind this equation, but that's just out of our focus, so let's simply accept this equation. With all the ingredients ready, now we can discuss the optimization algorithms.

### Gradient Descent
Well, the gradient descent is just too popular in this field (especially in optimizing neural networks), so I will not go deep in this topic. Furthermore, there are just too many variants of it, so this guy deserves its own post. Here, I just simply mention how it works: it just repeatedly descends along the gradient with a constant factor, hoping it will take us to a local minima. The vanilla gradient descent equation is as follows:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k).$$

where $$\alpha$$ is a learning rate.

### Newton-Raphson Method
Before talking about Gauss-Newton method, we first need to understand Newton-Raphson method. 

### Gauss-Newton Method

Now, Gauss-Newton method is a nonlinear least-squares optimization method. To apply this method, the system must be overdetermined, as the Jacobian of residual should be invertible. What is a `residual` ($$r$$)? It's a difference between the ground truth (target value) and the model's output.

$$r_{i} = y_{i} - f(x_{i}).$$