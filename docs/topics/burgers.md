<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Burgers Equation Solver

## Introduction

The Burgers equation is a non-linear Partial Differential Equation (PDE). It comes in two forms, the inviscid and viscous form, only determined by the presence of the diffusion term on the Right-Hand Side (RHS).

$$
\frac{\partial u}{\partial t}+u\frac{\partial u}{\partial x}=\nu\frac{\partial^2 u}{\partial x^2}
$$

The first term, $\frac{\partial u}{\partial t}$, is the unsteady term that allows us to integrate and find the conditions over various time steps. The second term, $u\frac{\partial u}{\partial x}$, is the convective or non-linear term. This gives the PDE its complex behavior. The final term, the only one on the RHS, $\nu\frac{\partial^2 u}{\partial x^2}$, is the dissipation term which allows the diffusion, which is like a "spreading", for the behavior. If one wants a good illustration, the heat equation, which isolates the diffusion behavior, is a good example and easy to implement [@CFD_anderson].

There is a slight modification that we can make to allow the computer to handle the non-linear term better. On one hand, the weak solution is popular for finitied difference or element methods. On the other hand, Lax offers an alternative form of the Burgers equation that contains a function of the velocity.

$$
\frac{\partial u}{\partial t}+\frac{\partial f}{\partial x}=\nu\frac{\partial^2 u}{\partial x^2}
$$

The eagle-eyed may notice that $u\frac{\partial u}{\partial x}=\frac{\partial }{\partial x}(\frac{u^2}{2})$, which allows us to define $f$ as $f=\frac{u^2}{2}$, referred to as the flux formulation. Now, to get the Burgers equation into a time-integration-friendly form,

$$
T(t)=\frac{\partial u}{\partial t}=\nu\frac{\partial^2 u}{\partial x^2}-\frac{\partial f}{\partial x}
$$

## Discretized Formulation

The derivative operators can be represented by a system of linear equations where there are matrices that act as the derivatives along the connected points. Thus,

$$
\mathbf{T}(t)_{i}=\nu[\mathbf{A}]\mathbf{u}_{i}-[\mathbf{B}]\mathbf{f}_{i}
$$

Where $[\mathbf{A}]=[\frac{\partial^2}{\partial x^2}_{i}]=\frac{1}{\Delta x^2}[\mathcal{G}(p,2)]$ and $[\mathbf{B}]=[\frac{\partial}{\partial x}_{i}]=\frac{1}{\Delta x}[\mathcal{G}(p,1)]$, and $\mathcal{G}(<points\:in\:stencil>,<derivative\:order>)$ as the numerical gradient calculated from a Taylor series.

## Residual Calculation

The implication of the discretized approximation is that there is some data that is not translated from the continuous to discretized. Fundamentally, the residual is the difference between the expected time integration and the actual time integration, i.e.:

$$
\mathbf{R}(t)_{i}=\left( LHS-RHS \right)^{p}=\left( \frac{\partial u}{\partial t}-\left( \nu\frac{\partial^2 u}{\partial x^2}-\frac{\partial f}{\partial x} \right) \right)^{p}
$$

Or

$$
\mathbf{R}(t)_{i}=\left( \mathbf{T}(t)_{i}-\left( \nu[\mathbf{A}]\mathbf{u}_{i}-[\mathbf{B}]\mathbf{f}_{i} \right) \right)^p
$$

Where $p$ is the order of the norm, typically $p=2$ for the L2 norm. For the perfect answer, the residual is obviously zero. However, due to discretization losses, this will not be the case.

