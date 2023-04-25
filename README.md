# Using physics-informed neural networks to solve the advection equation

## Background

Physics-informed neural networks (PINNs) were [introduced](https://www.sciencedirect.com/science/article/pii/S0021999118307125) by Raissi, Perdikaris and Karniadakis in 2019, as a method of finding numerical solutions to continuous and discrete-time partial differential equations, as well as parameterising those equations using data.  In this repository, I want to concentrate on the case of finding a numerical solution to the continuous-time advection equation.  No doubt this has been done my many other authors, but I'm trying to teach myself!

The advection equation in 1 spatial dimension is

$$
\frac{\partial u}{\partial t} = -\frac{\partial}{\partial x} (vu) \ .
$$

Here:

- $t$ is time
- $x$ is the spatial coordinate
- $u = u(t, x)$ is the thing we're trying to find.  It could represent temperature of a fluid, or concentration of a pollutant, for instance
- $v$ is the advection velocity.  It could be the velocity of the fluid, for instance.  Assume it is independent of $x$ and $t$.

The analytical solution of the advection equation is

$$
u = f(x - vt) \ ,
$$

where $u(0, x) = f(x)$ is the initial configuration of $u$ (the temperature distribution at $t=0$, for instance).  This is the solution that the PINN should produce.

The reason for choosing the advection equation is that its analytical solution is known and straightfoward, and that the advection equation is [quite tricky to solve using standard numerical techniques](https://mooseframework.inl.gov/modules/porous_flow/stabilization.html).  In naive schemes, the numerical solution suffers from over-shoots and under-shoots.  For instance, if the initial condition satisfies

$$
0 < f < 1 \ ,
$$

then at later times $u < 0$ and/or $u > 1$.  To get around this, many numerical schemes introduce "numerical diffusion", but if care is not taken, this means that after some time the numerical solution is just $u$ is constant, independent of the initial condition.  (An animation may be found [here](https://mooseframework.inl.gov/modules/porous_flow/numerical_diffusion.html).)  Of course, numerical techniques to get around these problems are known, but the advection equation is not as simple as expected.  Do PINNs suffer any difficulties?

## Problem description

Assume the spatial domain is bounded: $-1 \leq x \leq 1$.  Assume the initial condition is

$$u(0, x) = f(x) = \left\\{
\begin{array}{ll}
1 & x \leq 0 \\
0 & x > 0
\end{array}\right.
$$

Assume the boundary conditions are Dirichlet (fixed value) on the left:

$$
u(t, -1) = 1 \ ,
$$

and Neumann (fixed flux, which is zero in this case) on the right:

$$
\frac{\partial}{\partial x}u(t, 1) = 0 \ .
$$

Assume the velocity, $v = 1$, so the analytic solution is $u(t, x) = f(x - t)$.  This is shown in the following plots.

![Surface plot of solution](analytic_solution.png)

![Animated solution](analytic_solution.mp4)
