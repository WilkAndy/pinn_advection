##################################################################################
# This script performs the inverse problem for the advection equation.
# That is: given some observational data (in observations.csv):
#  - find the advection velocity
#  - and use a neural network to solve the advection equation
#
# As with advection.py, this code trains a neural network so that it solves
# the advection equation
# du/dt = - velocity * du/dx
# but here, the velocity is unknown, and is determined by comparison with observations
#
# The code is actually very similar to advection.py, but to minimise the loss,
# its derivatives with respect to velocity are needed (so that gradient descent can
# be used).  This necessitates explicitly defining the weights and biases and
# the neural network architecture, as well as the gradient-descent.  Hence, things
# like
#     model = Sequential()
#     model.add(Dense(...))
#     model.fit(...)
# expand to many lines of code!
##################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import tensorflow as tf
import pandas as pd
tf.keras.utils.set_random_seed(12)

velocity = tf.Variable(0.5, dtype = tf.float32) # advection velocity with initial guess = 0.5

##########################################################################
# Set up the (t, x) points on which to evaluate:
# - the observations
# - the Dirichlet conditions u(t, x=-1) = 1 = vals_dirichlet
# - the Neumann conditions du/dx(t, x=1) = 0 = flux_neumann
# - the advection PDE on interior points, du/dt + velocity * du/dx = 0
##########################################################################
num_dirichlet = 101
num_neumann = 102
num_interior = 10000
obs = pd.read_csv("observations.csv", comment = "#")
T_obs = tf.constant(obs['T'], dtype = tf.float32)
X_obs = tf.constant(obs['X'], dtype = tf.float32)
vals_obs = tf.constant(obs['u'], dtype = tf.float32)
T_dirichlet = tf.random.uniform(shape = [num_dirichlet], minval = 0, maxval = 1, dtype = tf.float32)
X_dirichlet = tf.constant([-1] * num_dirichlet, dtype = tf.float32)
vals_dirichlet = tf.constant([1] * num_dirichlet, dtype = tf.float32)
T_neumann = tf.random.uniform(shape = [num_neumann], minval = 0, maxval = 1, dtype = tf.float32)
X_neumann = tf.constant([1] * num_neumann, dtype = tf.float32)
flux_neumann = tf.constant([0] * num_neumann, dtype = tf.float32)
T_interior = tf.random.uniform(shape = [num_interior], minval = 0, maxval = 1, dtype = tf.float32)
X_interior = tf.random.uniform(shape = [num_interior], minval = -1, maxval = 1, dtype = tf.float32)

###############################################################################################################
# The code works by minimising the loss function, which is a weighted sum of:
# - loss_dirichlet operating on the "observational" points, which attempts to ensure that u - vals_obs = 0
# - loss_dirichlet operating on the "dirichlet" points, which attempts to ensure that u - vals_dirichlet = 0
# - loss_neumann operating on the "neumann" points, which attempts to ensure that du/dx - flux_neumann = 0
# - loss_de, which attempts to ensure that du/dt + velocity * du/dx = 0 at all the "interior" points
# The relative weighting of each term obviously impacts the outcome.
###############################################################################################################

def loss_dirichlet(t, x, u_desired):
    ''' Evaluate the observations or Dirichlet boundary condition (both are "fixed u" conditions), ie
    sum_over(t, x)(|u - u_desired|^2) / number_of_(t, x)_points, where u is given by the NN model
    '''
    u_vals = tf.reshape(model(tf.stack([t, x], 1)), [len(t)]) # "model" is the NN predicted value, given (t, x)
    return tf.reduce_mean(tf.square(u_vals - u_desired))

def loss_neumann(t, x, flux_desired):
    ''' Evaluate the Neumann boundary condition, ie
    sum_over(t, x)(|du/dx - flux_desired|^2) / number_of_(t, x)_points, where u is given by the NN model
    '''
    # First, use TensorFlow automatic differentiation to evaluate du/dx, at the points (t, x), where u is given by the NN model
    with tf.GradientTape(persistent = True) as tp:
        tp.watch(x)
        u = model(tf.stack([t, x], 1)) # "model" is the NN predicted value, u, given (t, x)
    u_x = tp.gradient(u, x)
    del tp
    # Now return the loss
    return tf.reduce_mean(tf.square(u_x - flux_desired))

def loss_de(t, x):
    ''' Returns sum_over_(t, x)(|du/dt + velocity * du/dx|^2) / number_of_(t, x)_points
    '''
    # First, use TensorFlow automatic differentiation to evaluate du/dt and du/dx, at the points (t, x), where u is given by the NN model
    with tf.GradientTape(persistent = True) as tp:
        tp.watch(t)
        tp.watch(x)
        u = model(tf.stack([t, x], 1)) # "model" is the NN predicted value, u, given (t, x)
    u_t = tp.gradient(u, t)
    u_x = tp.gradient(u, x)
    del tp
    # The loss is just the mean-squared du/dt + velocity * du/dx
    return tf.reduce_mean(tf.square(u_t + velocity * u_x))

@tf.function # decorate for speed
def loss():  
    # This function has no arguments, which is slightly different from advection.py
    # advection.py loss function needed arguments just because Keras/Tensorflow needed them
    weight_obs = 1
    weight_dirichlet = 1
    weight_neumann = 1
    weight_de = 1
    return weight_obs * loss_dirichlet(T_obs, X_obs, vals_obs) + weight_dirichlet * loss_dirichlet(T_dirichlet, X_dirichlet, vals_dirichlet) + weight_neumann * loss_neumann(T_neumann, X_neumann, flux_neumann) + weight_de * loss_de(T_interior, X_interior)

##################################################################################
# The following rather large block of code defines the weights and biases as
# a whole lot of tf.Variables.  This is in contrast to advection.py, where
# their definition was hidden inside the lines
# model = Sequential()
# model.add(Dense(...))
# The weights are Glorot initialised and the biases are initialised to zero
# The reason the weights and biases are explicitly defined is that later
# they'll appear in the explicit definition of the neural net.  They are
# also collected into the "params" list, which appears in d(loss)/d(params)
# that will be used in the gradient-descent algorithm
##################################################################################
depth = 5           # depth of NN
width = 10          # width of fully-connected NN
def glorot_init_weight(in_dim, out_dim):
    return tf.Variable(tf.random.truncated_normal(shape = [in_dim, out_dim], mean = 0.0, stddev = np.sqrt(2.0 / (in_dim + out_dim)), dtype = tf.float32), dtype = tf.float32)
def zero_init_bias(in_dim, out_dim):
    return tf.Variable(tf.zeros(shape = [in_dim, out_dim], dtype = tf.float32), dtype = tf.float32)
weights = [] # all the weights
biases = [] # all the biases
params = [] # all the weights and biases AND THE VELOCITY!
# 2 inputs = (t, x)
w = glorot_init_weight(2, width)
b = zero_init_bias(1, width)
weights.append(w)
biases.append(b)
params.append(w)
params.append(b)
for d in range(1, depth):
    w = glorot_init_weight(width, width)
    b = zero_init_bias(1, width)
    weights.append(w)
    biases.append(b)
    params.append(w)
    params.append(b)
# 1 output =  u
w = glorot_init_weight(width, 1)
b = zero_init_bias(1, 1)
weights.append(w)
biases.append(b)
params.append(w)
params.append(b)
# NOTE!!  Here the velocity is added to params.  This is so that
# d(loss)/d(params) will include derivatives with respect to velocity
# and hence the gradient descent will try to find the best velocity
# (the value that minimises the loss)
params.append(velocity)

#######################################################################
# This block of code defines the neural network explicitly.  It
# takes the place of the lines
#   model = Sequential()
#   model.add(Dense(...))
# that appear in advection.py.  It is necessary to define the NN
# explicitly, so that the weights and biases defined above appear
# in the NN, so that d(loss)/d(params) can be used in the gradient
# descent
#######################################################################
def model(x):
    # This uses the relu activation function.  relu(x) = max(x, 0)
    # Some other alternatives like tanh, relu and softplus could be used
    z = x
    for d in range(depth):
        w = weights[d]
        b = biases[d]
        zp = tf.add(tf.matmul(z, w), b)
        #z = tf.math.log(1 + tf.math.exp(zp))          # softplus activation
        #z = tf.math.tanh(zp)                          # tanh activation
        #z = tf.where(zp > 0, zp, tf.math.exp(zp) - 1) # elu activation
        z = tf.math.maximum(zp, 0)                     # relu activation
    w = weights[depth]
    b = biases[depth]
    zp = tf.add(tf.matmul(z, w), b)
    return tf.math.maximum(zp, 0) # might like to change this activation function if the hidden layers are also changed

################################################################
# The following does one step of the gradient descent
# Note the appearance of d(loss)/d(params), which will
# mean the params (weights, biases AND velocity) will be
# altered so as to reduce the loss
################################################################
optimizer = tf.keras.optimizers.Adam()
@tf.function # decorate for speed
def gradient_descent():
    with tf.GradientTape(persistent = True) as tp:
        epoch_loss = loss()
    gradient = tp.gradient(epoch_loss, params) # d(loss)/d(params)
    del tp
    # because params includes weights, biases and velocity, the following line
    # alters all of these to reduce the loss
    optimizer.apply_gradients(zip(gradient, params))
    return epoch_loss

################################################################
# Do the gradient descent
################################################################
epochs = 1000 # training epochs
for epoch in range(epochs):
    epoch_loss = gradient_descent()
    print("epoch =", epoch, "loss =", epoch_loss.numpy(), "velocity =", velocity.numpy())

####### The remainder of the code is basically identical to advection.py
#
# Output some informative information 
print("After training, the losses are:")
print("Observations = ", loss_dirichlet(T_obs, X_obs, vals_obs))
print("Dirichlet = ", loss_dirichlet(T_dirichlet, X_dirichlet, vals_dirichlet))
print("Neumann = ", loss_neumann(T_neumann, X_neumann, flux_neumann))
print("DE = ", loss_de(T_interior, X_interior))
with open("observations.csv", "r") as f:
    true_velocity = float(f.readline().strip().split("=")[1])
print("The velocity is predicted to be", velocity.numpy(), "with true velocity ", true_velocity)

# Display the results graphically
fig = plt.figure()
axis = plt.axes(xlim = (-1, 1), ylim = (-0.1, 1.1))
line, = axis.plot([], [], linewidth = 2)
stuff_to_animate = [axis.plot([], [], linewidth = 2, color = 'k', label = 'True, v = ' + str(true_velocity))[0], axis.plot([], [], linewidth = 2, color = 'r', linestyle = '--', label = 'PINN, predicted v = ' + str(round(velocity.numpy(), 2)))[0], axis.annotate(0, xy = (-0.9, 0.9), xytext = (-0.9, 0.9), fontsize = 13)]
def init():
    stuff_to_animate[0].set_data([], [])
    stuff_to_animate[1].set_data([], [])
    stuff_to_animate[2].set_text("t = 0.00")
    return stuff_to_animate
xdata = tf.constant(np.linspace(-1, 1, 1000), dtype = tf.float32)
def front(x):
    ''' The shape of the front that is advected.  This is what was used in
    observations.py to create observations.csv.  It is animated in the
    matplotlib lines below for comparison with the neural net's prediction
    '''
    window = 0.2
    return tf.where(x <= -1, 1, tf.where(x >= -1 + 2 * window, 0, 0.5 + (x + 1 - window) * (tf.pow(x + 1 - window, 2) - 3 * tf.pow(window, 2)) / 4 / tf.pow(window, 3)))
def animate(i):
    t = 0.01 * i
    tdata = tf.constant([t] * len(xdata), dtype = tf.float32)
    stuff_to_animate[0].set_data(xdata, front(xdata - true_velocity * t))
    stuff_to_animate[1].set_data(xdata, model(tf.stack([tdata, xdata], 1)))
    stuff_to_animate[2].set_text("t = " + str(round(t, 2)))
    return stuff_to_animate
anim = ani.FuncAnimation(fig, animate, init_func = init, frames = 101, interval = 20, blit = True)
plt.grid()
plt.xlabel("x")
plt.ylabel("u")
plt.legend(loc = 'upper right')
plt.title("Inverse problem: true and PINN prediction")
anim.save('inverse.gif', fps = 30)
plt.show()
sys.exit(0)
