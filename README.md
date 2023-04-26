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

then at later times the numerical solution breaks these bounds ($u < 0$ and/or $u > 1$).  To get around this, many numerical schemes introduce "numerical diffusion", but if care is not taken, this means that after some time the numerical solution is $u = $ constant, independent of the initial condition.  (An animation may be found [here](https://mooseframework.inl.gov/modules/porous_flow/numerical_diffusion.html).)  Of course, numerical techniques to get around these problems are known, but the advection equation is not as simple as expected.  Do PINNs suffer any difficulties?

## Required packages

Python with [TensorFlow](https://www.tensorflow.org/) is used to solve the PINNs in this repository.  To run the python scripts, `matplotlib` and `tensorflow` are needed.  Install them using, for instance,

```
conda install -c conda-forge tensorflow
conda install -c conda-forge matplotlib
```

## A primer: integrating a function

As a primer for the advection problem, consider building a neural network to find $u$, where the derivative of $u$ is known.  That is, given $f(x)$, find $u(x)# such that

$$
\frac{\mathrm{d}u}{\mathrm{d}x} = f(x) \ .
$$

This means the neural network is integrating the function $f$.  In addition, assume that $u(0) = u_{0}$, where $u_{0}$ is specified, which fixes the constant of integration.  For ease of presentation, assume that $0 \leq x$.  Note that these two conditions are directly analogous to the general PDE situation solved by PINNs, in particular:

- $\mathrm{d}u/\mathrm{d}x = f$ is equivalent to the PDE
- $u(0) = u_{0}$ is equivalent to the boundary and initial conditions.

This problem is different than usual nonlinear regression using a neural network.  In the usual case, the values, $u$, are known at certain $x$ points, and the neural network is trained to produce these values.  Nevertheless, most of the usual neural-network architecture can be used for this integration problem.  All that is required is to build an appropriate loss function, and let the usual netural-network machinery find the solution.

The code to perform the integration is in [integrate.py](integrate.py).

The loss function is a linear combination of the conditions $\mathrm{d}u/\mathrm{d}x - f = 0$ and $u(0) - u_{0} = 0$.  The former uses TensorFlow's automatic differentiation (in this code snippet, `x` are the points on the domain at which the `function_values` ($f$) are known):

```
def loss_de(x, function_values):
    ''' Returns sum_over_x(|du/dx - function_values|^2) / number_of_x_points
    '''
    # First, use TensorFlow automatic differentiation to evaluate du/dx, at the points x, where u is given by the NN model
    with tf.GradientTape(persistent = True) as tp:
        tp.watch(x)
        u = model(x) # "model" is the NN's value
    u_x = tp.gradient(u, x)
    del tp
    # The loss is just the mean-squared du/dx - function_values
    return tf.reduce_mean(tf.square(u_x - function_values))
```

The constraint $u(0) - u_{0} = 0$ is encoded as (in this code snippet, `x` are the points on the domain, recall $0 \leq x$) :

```
def loss_bdy(x, val_at_zero):
    ''' Evaluate the boundary condition, ie u(0) - val_at_zero, where u is given by the NN model
    '''
    val = model(tf.convert_to_tensor([x[0]], dtype = tf.float32)) - val_at_zero # "model" is the NN predicted value, given x
    return tf.reduce_mean(tf.square(val))
```

Finally, the loss used in the neural-network training process is a weighted linear combination of these (in this code snippet, `X` are the points on the domain at which the `fcn` values ($f$) are known, and `value_at_0` is $u_{0}$):

```
@tf.function # decorate for speed
def loss(ytrue, ypred):
    ''' The loss used by the training algorithm.  Note that ytrue and ypred are not used,
    but TensorFlow specifies these arguments
    '''
    bdy_weight = 1
    de_weight = 1
    return bdy_weight * loss_bdy(X, value_at_0) + de_weight * loss_de(X, fcn)
```

The following figure shows the result for $f(x) = \cos(x)$ and $u_{0} = 0.7$:

![Result of integrating a function](integrate.png)

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

![Animated solution](analytic_solution.gif)

## The PINN

The PINN must have two inputs, $(t, x)$, and one output, $u$.  Raissi, Perdikaris and Karniadakis state that the PINN must be of sufficient complexity to be able to represent the solution, $u$.


