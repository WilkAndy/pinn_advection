# Using physics-informed neural networks to solve the advection equation

## Background

Physics-informed neural networks (PINNs) were [introduced](https://www.sciencedirect.com/science/article/pii/S0021999118307125) by Raissi, Perdikaris and Karniadakis in 2019, as a method of finding numerical solutions to continuous and discrete-time partial differential equations, as well as parameterising those equations using data.  In this repository, I want to concentrate on the case of finding a numerical solution to the continuous-time advection equation.  No doubt this has been done my many other authors, but I'm trying to teach myself!

The advection equation in 1 spatial dimension is

$$
\frac{\partial u}{\partial t} = -\frac{\partial}{\partial x} vu \ .
$$


