---
title: Nonlinear Optimization Methods
date: 2024-08-15 20:52:00 +0900
categories: [theory, mathematics]
tags: [gradient descent, gauss newton, levenberg marquardt, lm, nonlinear, least squares, optimization]     # TAG names should always be lowercase
description: Nonlinear least squares optimization methods used in computer vision problems.
math: true
---

### Introduction

I mean, you can simply use `scipy.optimize`, `torch`, or `ceres-solver`, to solve a nonlinear optimization problem. So, why bother studying these?
It's because as a *professional* computer vision engineer, you sometimes need to tackle the core of these optimization techniques: maybe your team implemented their own optimization library, and you need to handle errors in it (true story). To do so, the underlying, fundamental knowledge about this topic is required.

Now, there are basically 3 optimization techniques that we commonly encounter in this field: gradient descent (and a whole bunch of its variants - I will not talk much about this guy in this post), Gauss-Newton method, and Levenberg-Marquardt (LM) method. I've seen someone using `BFGS`, but I'd argue that it's really a rare case in this field.

Okay, cool, but, what really is the `gradient`? 
What is the concrete, mathematical definition of `Jacobian` or `Hessian` that we've heard while learning these methods?
Or better yet, what is the least squares optimization problem in the first place?

### Definitions

#### Least Squares Problem
So the least squares problem is to find the optimal (learnable) parameters that minimize the sum of the squares of the `residuals`. What are the `residuals`  then? It's just a bunch of differences between the ground truth (target value) and the model's output at the corresponding point:

$$r_{i} = y_{i} - f(x_{i}).$$

Now, you can clearly see these individual residuals, when collected as a whole, form a **vector**: $$(r_{1}, r_{2}, ..., r_{n})$$. Well, I mean, each residual could have been a vector already, but, let's not take that into account. Because our optimization target is the *sum of the squares of the residuals*:

$$S = \sum_{i=1}^{n} r_{i}^{2}.$$

it is now a **scalar**, even if each individual residual was actually a vector (and it's again just the sum of all its elements). So, when we are solving least-squares optimization problem, we are essentially optimizing our parameters based on a **scalar-valued function**. In other words, this scalar function acts as a projection of the information of what we want our model to fit to.

#### Gradient
Now let's see what the mathematical definition of these *technical* terms. Mathematically, gradient is defined over a **scalar-valued** differentiable function $$f$$ that depends on $$n$$ variables (note that the $$f$$ and $$n$$ here has nothing to do with the previous ones). The gradient $$∇f$$ is then a vector field **(vector-valued function)**, that represents the direction and the rate of steepest slope of $$f$$ at a given point. You know, it's just an array of partial derivatives of $$f$$.

$$∇f_{i} = \frac{\partial {f}}{\partial{x_i}},$$

$$∇f: \mathbb{R}^n \to \mathbb{R}^n, f: \mathbb{R}^n \to \mathbb{R}.$$

For convenience though, let's treat the gradient vector as a column vector: $$∇f \in \mathbb{R}^{n\times 1}$$.

#### Jacobian
Jacobian on the other hand, is defined over a **vector-valued** differentiable function $$\boldsymbol{f}$$ (notice the notation is now bold). As was in gradient, it's just an array of all its partial derivatives, but this time it forms a matrix, because the output domain is not a scalar but a vector. Say that the $$\boldsymbol{f}$$ depends on $$n$$ variables, and it maps those variables to an $$m$$ dimensional vector. Then, the Jacobian $$J$$ of $$\boldsymbol{f}$$ is an $$m\times n$$ matrix. 
In other words, it's just a list of gradients of length $$m$$, where each gradient corresponds to each scalar element of the $$m$$ dimensional output vector.


$$J_{i,j} = \frac{\partial \boldsymbol{f}_i}{\partial{x_j}},$$

$$J \in \mathbb{R}^{m \times n}, \boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m.$$

#### Hessian
Hessian is again defined over a **scalar-valued** function $$f$$ depending on $$n$$ variables. The Hessian matrix $$H$$ of $$f$$ is a square matrix of shape $$n\times n$$ and it represents all the second-order partial derivatives of $$f$$, describing the local curvature of $$f$$. FYI, if all the second partial derivatives are continuous, then the $$H$$ is symmetric (well, so mathematicians say).


$$H_{i,j} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},$$

$$H \in \mathbb{R}^{n \times n}, f: \mathbb{R}^n \to \mathbb{R}.$$

Also note that the Hessian is equal to the jacobian of the gradient of $$f$$:

$$\mathbf{H}(f(\mathbf{x})) = \mathbf{J}(\nabla f(\mathbf{x})).$$

#### Taylor Expansion of a Multivariable Scalar Function

The second-order Taylor expansion of a multivariable scalar function $$f$$, can be written compactly using Jacobian and Hessian of $$f$$. That is:

$$y = f(\mathbf{x} + \Delta \mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^{\mathrm{T}} \Delta \mathbf{x} + \frac{1}{2} \Delta \mathbf{x}^{\mathrm{T}} \mathbf{H}(\mathbf{x}) \Delta \mathbf{x}.$$

I wish I could prove all the details of the mathematics behind this equation, but that's just out of our focus, so let's simply accept this equation. With all the ingredients ready, now we can discuss the optimization algorithms.

### Gradient Descent
Well, the gradient descent is just too popular in this field (especially in optimizing neural networks), so I will not go deep in this topic. Furthermore, there are just too many variants of it, so this guy deserves its own post. Here, I just simply mention how it works: it just repeatedly descends along the gradient with a constant factor, hoping it will take us to a local minima. The vanilla gradient descent equation is as follows:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k).$$

where $$\alpha$$ is a learning rate.

### Newton-Raphson Method
Before talking about Gauss-Newton method, we first need to understand Newton-Raphson method. The Newton-Raphson method is not a nonlinear optimization method, but it's a method to find a root of a scalar function $$f$$ (a solution of $$f(x) = 0$$). For optimization though, this method can be applied onto $$∇f$$, to find a critical point of $$f$$. 

The original Newton-Raphson method for a single-variable, scalar function looks like this:

$$x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}.$$

For the generalization to a multivariable, vector-valued function $$\boldsymbol{f}$$, a neat formulation can only exist when the input dimension is the same as the output dimension ($$\boldsymbol{f}: \mathbb{R}^k \to \mathbb{R}^k$$):

$$\mathbf{x}_{n+1} = \mathbf{x}_{n} - J_{\boldsymbol{f}}(\mathbf{x}_{n})^{-1} \boldsymbol{f}(\mathbf{x}_{n}).$$

It is because for a Jacobian to be invertible, it should be a square matrix. What about pseudo-inverse for a non-square Jacobian? Good question! This generalization actually leads us to the Gauss-Newton method. But for now, let's focus on the Newton Raphson method.

For a least-squares optimization purpose, we are only concerned of a scalar-valued function $$f$$, but we are applying the method onto $$∇f$$, which is now a vector-valued function. Then, the method would go like:

$$\mathbf{x}_{n+1} = \mathbf{x}_{n} - J_{∇f}(\mathbf{x}_{n})^{-1} ∇f(\mathbf{x}_{n}).$$

However, we mentioned before that the Jacobian of the gradient of $$f$$ is equal to the Hessian of $$f$$. Therefore, the nonlinear optimization using Newton-Raphson method can be expressed using the Hessian of $$f$$ as follows.

$$\mathbf{x}_{n+1} = \mathbf{x}_{n} - H_{f}(\mathbf{x}_{n})^{-1} ∇f(\mathbf{x}_{n}).$$

### Gauss-Newton Method

Now, Gauss-Newton method is a nonlinear least-squares optimization method. To apply this method, the system must be overdetermined, as the Jacobian of residual should be invertible. 