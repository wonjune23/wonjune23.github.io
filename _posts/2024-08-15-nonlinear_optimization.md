---
title: Fundamentals of Nonlinear Optimization Methods
date: 2024-09-17 01:15:00 +0900
categories: [theory, mathematics]
tags: [gradient descent, gauss newton, levenberg marquardt, lm, nonlinear, least squares, optimization]     # TAG names should always be lowercase
description: Mathematical foundations behind the nonlinear least squares optimization methods used in computer vision problems.
math: true
---

### Introduction

There are various tools like `scipy.optimize`, `torch`, or `ceres-solver`, to solve a nonlinear optimization problem. So, why bother studying the underlying mathematics?
It's because as a *professional* computer vision engineer, you sometimes need to tackle the core of these optimization techniques: maybe your team implemented their own optimization library, and *you* are the one who needs to handle errors in it (true story). To do so, the underlying, fundamental knowledge about the topic is required.

Now, there are basically three optimization techniques that we commonly encounter in computer vision problems: gradient descent (and a whole bunch of its variants - I will not talk much about this guy in this post), Gauss-Newton method, and Levenberg-Marquardt (LM) method. I've seen someone using other methods, but I'd argue that it's really a rare case in this field.

Okay, cool, but, what really is the `gradient`? 
What is the concrete, mathematical definition of `Jacobian` or `Hessian` that we've heard while learning these methods?
Or better yet, what is the least squares optimization problem in the first place?

### Definitions

#### Least Squares Problem
So the least squares problem is to find the optimal (learnable) parameters that minimize the sum of the squares of the `residuals`. What are the `residuals`  then? It's just a bunch of differences between the ground truth (target value) and the model's output at the corresponding point:

$$r_{i} = y_{i} - f(x_{i}).$$

Now, you can clearly see these individual residuals, when collected as a whole, form a **vector**: $$(r_{1}, r_{2}, ..., r_{n})$$. Well, I mean, each residual could have been a vector already (as opposed to being a scalar), but, let's not take that into account. Because our optimization target is the *sum of the squares of the residuals*:

$$S = \sum_{i=1}^{n} r_{i}^{2}.$$

it is now a **scalar**, even if each individual residual was actually a vector. So, when we are solving least-squares optimization problem, we are essentially optimizing our parameters based on a **scalar-valued function**. In other words, this scalar function acts as a projection of the information of what we want our model to fit to.

#### Gradient
Now let's see what the mathematical definition of these *technical* terms. Mathematically, gradient is defined over a **scalar-valued** differentiable function $$f$$ that depends on $$n$$ variables (note that the $$f$$ and $$n$$ here has nothing to do with the previous ones - I will abuse notations a lot!). The gradient $$∇f$$ is then a vector field **(vector-valued function)**, that represents the direction and the rate of steepest slope of $$f$$ at a given point. You know, it's just an array of partial derivatives of $$f$$.

$$∇f_{i} = \frac{\partial {f}}{\partial{x_i}},$$

$$∇f: \mathbb{R}^n \to \mathbb{R}^n, f: \mathbb{R}^n \to \mathbb{R}.$$

For convenience though, let's treat the gradient vector as a column vector: $$∇f \in \mathbb{R}^{n\times 1}$$.

#### Jacobian
Jacobian on the other hand, is defined over a **vector-valued** differentiable function $$\boldsymbol{f}$$. As was in gradient, it's just an array of all its partial derivatives, but this time it forms a matrix, because the output domain of $$\boldsymbol{f}$$ is not a scalar but a vector. Say that the $$\boldsymbol{f}$$ depends on $$n$$ variables, and it maps those variables to an $$m$$ dimensional vector. Then, the Jacobian $$J$$ of $$\boldsymbol{f}$$ is an $$m\times n$$ matrix. 
In other words, it's just a list of gradients of length $$m$$.


$$J_{ij} = \frac{\partial \boldsymbol{f}_i}{\partial{x_j}},$$

$$J \in \mathbb{R}^{m \times n}, \boldsymbol{f}: \mathbb{R}^n \to \mathbb{R}^m.$$

#### Hessian
Hessian is again defined over a **scalar-valued** function $$f$$ depending on $$n$$ variables. The Hessian matrix $$H$$ of $$f$$ is a square matrix of shape $$n\times n$$ and it represents all the second-order partial derivatives of $$f$$, describing the local curvature of $$f$$. FYI, if all the second partial derivatives are continuous, then the $$H$$ is symmetric (well, so mathematicians say).


$$H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},$$

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

#### Derivations

Now, Gauss-Newton method is a nonlinear least-squares optimization method. This method can be derived from the Newton-Raphson method. For least-squares optimization, the $$f = \sum_{i=1}^{n} r_{i}^{2}$$, and thus the gradient is:

$$∇f_{i} =2\sum _{j=1}^{n}r_{j}{\frac {\partial r_{j}}{\partial x_{i}}}.$$

or, using matrix notation, (where vectors are treated as column vectors):

$$∇f = 2J_{r}^{\operatorname{T}}r.$$

Again, the Hessian is equal to jacobian of gradient (or, just look at the definition of Hessian, it's just a matrix of all the second-order partial derivatives). Therefore:

$$H_{ij}=2\sum _{k=1}^{n}\left({\frac {\partial r_{k}}{\partial x_{i}}}{\frac {\partial r_{k}}{\partial x_{j}}}+r_{k}{\frac {\partial ^{2}r_{k}}{\partial x_{i}\partial x_{j}}}\right).$$

Now, if we approximate the Hessian by ignoring the second-order derivative term (assuming the second-order term is negligible), we have:

$$H_{ij}\approx 2\sum _{k=1}^{n}J_{ik}J_{jk},$$ 

using matrix notation, it is expressed as: 

$$H\approx 2J_{r}^{\operatorname{T}}J_{r}.$$


Now, look at the Newton-Raphson method for least-squares optimization (the one that involves $$H$$ and $$∇f$$). If we substitute these expressions we get the Gauss-Newton method:

$$\mathbf{x}_{n+1}=\mathbf{x}_{n}-\left(J_{r}^{\operatorname{T}}J_{r}\right)^{-1}J_{r}^{\operatorname{T}}r\left(\mathbf{x}_{n}\right).$$

Therefore, the Gaussn-Newton method is nothing but a Newton-Raphson method applied to the gradient of $$f$$, where the $$H$$ is approximated by ignoring its second-order derivative terms. 

At this point, I must also mention another interpretation of the Gauss-Newton method. The residual function can be linearly approximated such that: 

$$r_{n+1} \approx r_{n} + J_{r_{n}} \Delta$$ 

Where $$\Delta = x_{n+1} - x_{n}$$. 

As we should minimize the sum of the approximated residuals in terms of $$\Delta$$:

$$\operatorname*{argmin}_{\Delta} \left\{ \sum_{i=1}^{n} \left( r_i + (J_{r} \Delta)_i \right)^2 \right\}$$

This is a linear least-squares problem that can be solved in a closed form. Note that it is a quadratic approximation of the cost function (sum of squared residuals) in terms of $\Delta$. Because it is of quadratic form, it is larger than or eqaual to zero. We thus set it equal to zero to find the best solution using the pseudo-inverse. It gives the same Gauss-Newton equation (Note that the expression $$(J_{r}^{\operatorname{T}}J_{r})^{-1}J_{r}^\operatorname{T}$$ is actually a pesudo-inverse of the $${J_{r}}$$).

Therefore, the Gauss-Newton method can be interpreted in two different ways: 

1. Linear approximation of the residual function (or, quadratic approximation of the cost function) and solving it in a closed form.
2. Applying Newton-Raphson method onto the gradient of the residual function, while linearly approximating the Hessian matrix.

This is why we often see different interpretations of the method: some say it's a quadratic approximation, others says it's a linear approximation, but all of them are valid in their own ways. Also note that this is an iterative method, and the equation (and thus the approximation as well) is applied iteratively.

#### Analysis

Now, let's see the properties of the method. The first thing we notice is that it does not contain the Hessian computation, but it only requires to compute a Jacobian. This saves a lot of computational resource, making this method much more efficient than the Newton-Raphson method. 

Also, the system must not be underdetermined, because otherwise, the $$(J_{r}^{\operatorname{T}}J_{r})$$ term gets singular, and thus is not invertible. In other words, the Gauss-Newton method is an effective method for solving an overdetermined system (just like when we're bundle adjusting!).
Another thing to keep in mind: just like Newton-Raphson method, when the initial point is not close enough, or when the $$(J_{r}^{\operatorname{T}}J_{r})$$ is ill-conditioned, the Gauss-Newton method might not converge at all.

The Gauss-Newton method is quite a dangerous method to use in a real engineering problem, because there is nothing preventing from the $$(J_{r}^{\operatorname{T}}J_{r})$$ term being singular. If you try to perform *e.g.,* `np.linalg.inv(J.T @ J)`, it can raise an error, and if you didn't handle this exception beforehands, your system will go down! I mean, even if you did, there is just not much to do with this method if you can't get the inverse.

### Levenberg-Marquardt Method

Levenberg-Marquardt (LM) method, is the de-facto standard for solving 3D geometric computer vision problems like bundle adjustment. The algorithm adds a damping factor, $\lambda I$ $(\lambda > 0)$, to the Gauss-Newton equation:

$$\mathbf{x}_{n+1}=\mathbf{x}_{n}-(J_{r}^{\operatorname{T}}J_{r} + \lambda I)^{-1}J_{r}^{\operatorname{T}}r\left(\mathbf{x}_{n}\right).$$

As $\lambda$ increases, it behaves more like the gradient descent (remember, $J_{r}^{\operatorname{T}}r\left(\mathbf{x}_{n}\right)$ is proportional to the gradient of $f$), and otherwise, it behaves like the Gauss-Newton method.

As of the value of $\lambda$, it is a parameter that is really up to a user, but normally it is chosen dynamically, mimicking trust region methods. Briefly speaking: if the current $\lambda$ does not reduce the overall cost function, you can either increase or decrease it based on a predefined criteria. The most common method is to set a pre-defined multiplication factor $\nu$ $(\nu > 0)$, and you multiply it to $\lambda$, or divide by it, until you can find an appropriate value that efficiently reduces the cost function.

One important property of this method is that, while $J_{r}^{\operatorname{T}}J_{r}$ is **positive semi-definite** (by construction!), $(J_{r}^{\operatorname{T}}J_{r} + \lambda I)$ is actually **positive definite** (as opposed to being semi-). This means the resulting dampened matrix is always a full ranked matrix and is always invertible (why is that so? see <https://chatgpt.com/share/66ec6a5b-4c7c-800c-996f-9145450c9d65>). Now, `np.linalg.inv(J.T @ J + lambda * I)` will never raise `NotInvertibleError`!

Also note that the damping factor is not always $\lambda I$, but it can be something like $\lambda * diag(J_{r}^{\operatorname{T}}J_{r})$, for faster convergence.

### Conclusion

In this post, I made a brief overview of the mathematical concepts behind the gradient descent, Gauss-Newton, and Levenberg-Marquardt methods. I focused on the basic definitions of the mathematical terms and how the logics flow with those concepts. Personally, it was always tricky for me to comprehend the formulas when I had to use them, but now that I wrote this post, I could establish a concrete comprehension of the equations and the logics behind the methods. Wish it helps others who read this post as well!