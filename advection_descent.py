##################################################################################
# This does the same thing as advection.py, but it uses gradient descent
# with derivatives of the neural-net's loss with respect to its internal
# parameters (weights and biases) being explicitly calculated using
# automatic differentiation.
# This makes the code substantially longer, because the weights and biases
# have to be defined explicitly instead of implicitly via the Keras version
# model = Sequential(), etc
# The neural network has to be defined explicitly too.
# Finally, the gradient descent is explicitly written instead of just using
# Keras's Sequential.fit() 
# The comments below document the differences between this script and advection.py
#
# This code trains a neural network so that it solves the advection equation
# du/dt = - velocity * du/dx
# The initial condition is
#                | 1     if x <= -1
# u = front(x) = | cubic if -1 < x < 1 + 2 * window
#                | 0     if x >= 1 + 2 * window
# where the cubic is chosen so u is differentiable.  As "window" tends to zero
# this becomes a step function.
# The boundary conditions are u(x=-1) = 1, and du/dx(x=1) = 0
##################################################################################

####### From here until mentioned below, the code is identical to advection.py
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import tensorflow as tf
tf.keras.utils.set_random_seed(12)

def front(x):
    ''' The shape of the front that is advected.  The return values could
    represent temperature or concentration of a solute, for example, as a
    function of the input x.
    The functional form that is used below is:
       1 if x <= -1
       cubic if -1 < x < -1 + 2 * window
       0 if x >= -1 + 2 * window
    The cubic is chosen such that the result is differentiable
    '''
    # Note, ensure that -1 + 2 * window + velocity < 1, otherwise pulse will reach right-hand (no flux) boundary.  In practice -1 + 2 * window + velocity < 0.7 ensures easier convergence
    window = 0.2
    return tf.where(x <= -1, 1, tf.where(x >= -1 + 2 * window, 0, 0.5 + (x + 1 - window) * (tf.pow(x + 1 - window, 2) - 3 * tf.pow(window, 2)) / 4 / tf.pow(window, 3)))
    
velocity = 1.2      # advection velocity

##########################################################################
# Set up the (t, x) points on which to evaluate:
# - the initial condition u(t=0, x) = vals_initial
# - the Dirichlet conditions u(t, x=-1) = 1 = vals_dirichlet
# - the Neumann conditions du/dx(t, x=1) = 0 = flux_neumann
# - the advection PDE on interior points, du/dt + velocity * du/dx = 0
##########################################################################
num_initial = 10000
num_dirichlet = 101
num_neumann = 102
num_interior = 10000
T_initial = tf.constant([0] * num_initial, dtype = tf.float32)
X_initial = tf.random.uniform(shape = [num_initial], minval = -1, maxval = 1, dtype = tf.float32)
vals_initial = front(X_initial)
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
# - loss_dirichlet operating on the "initial" points, which attempts to ensure that u - vals_initial = 0
# - loss_dirichlet operating on the "dirichlet" points, which attempts to ensure that u - vals_dirichlet = 0
# - loss_neumann operating on the "neumann" points, which attempts to ensure that du/dx - flux_neumann = 0
# - loss_de, which attempts to ensure that du/dt + velocity * du/dx = 0 at all the "interior" points
# The relative weighting of each term obviously impacts the outcome.
###############################################################################################################

def loss_dirichlet(t, x, u_desired):
    ''' Evaluate the initial condition or Dirichlet boundary condition (both are "fixed u" conditions), ie
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

####### The code above here is identical to advection.py
@tf.function # decorate for speed
def loss():  
    ####### This function has no arguments, which is slightly different from advection.py
    ####### advection.py loss function needed arguments just because Keras/Tensorflow needed them
    weight_initial = 1
    weight_dirichlet = 1
    weight_neumann = 1
    weight_de = 1
    return weight_initial * loss_dirichlet(T_initial, X_initial, vals_initial) + weight_dirichlet * loss_dirichlet(T_dirichlet, X_dirichlet, vals_dirichlet) + weight_neumann * loss_neumann(T_neumann, X_neumann, flux_neumann) + weight_de * loss_de(T_interior, X_interior)

depth = 5           # depth of NN
width = 10          # width of fully-connected NN
activation = 'relu' # alternatives 'selu', 'softplus', 'sigmoid', 'tanh', 'elu', 'relu', but these ARE NOT CODED
epochs = 1000       # training epochs

####### Here are the major differences between advection.py and this script
####### In the code below, the weights and biases are explicitly defined and initialised
####### and collected into the weights, biases and params lists for later use
####### Note that they are tf.Variables, which means TensorFlow can calculate
####### derivatives of the neural network loss with respect to them
def glorot_init_weight(in_dim, out_dim, name):
    return tf.Variable(tf.random.truncated_normal(shape = [in_dim, out_dim], mean = 0.0, stddev = np.sqrt(2.0 / (in_dim + out_dim)), dtype = tf.float32), dtype = tf.float32, name = name)
def zero_init_bias(in_dim, out_dim, name):
    return tf.Variable(tf.zeros(shape = [in_dim, out_dim], dtype = tf.float32), dtype = tf.float32, name = name)
weights = []
biases = []
params = []
# 2 inputs = (t, x)
w = glorot_init_weight(2, width, "weight0")
b = zero_init_bias(1, width, "bias0")
weights.append(w)
biases.append(b)
params.append(w)
params.append(b)
for d in range(1, depth):
    w = glorot_init_weight(width, width, "weight" + str(d))
    b = zero_init_bias(1, width, "bias" + str(d))
    weights.append(w)
    biases.append(b)
    params.append(w)
    params.append(b)
# 1 output =  u
w = glorot_init_weight(width, 1, "weight" + str(depth))
b = zero_init_bias(1, 1, "bias" + str(depth))
weights.append(w)
biases.append(b)
params.append(w)
params.append(b)

####### The second major difference between advection.py and this code is
####### that the neural network model is explicitly written out, instead of
####### just using Keras's Sequential() construction
def model(x):
    z = x
    for d in range(depth):
        w = weights[d]
        b = biases[d]
        z = tf.math.maximum(tf.add(tf.matmul(z, w), b), 0) # relu activation
    w = weights[depth]
    b = biases[depth]
    return tf.math.maximum(tf.add(tf.matmul(z, w), b), 0) # relu activation

optimizer = tf.keras.optimizers.Adam()

####### The third major difference is that gradient descent is explicitly
####### written out here instead of using Sequential.fit() that is employed
####### in advection.py.   This is the whole point of this script!!
@tf.function
def gradient_descent():
    with tf.GradientTape(persistent = True) as tp:
        epoch_loss = loss()
    gradient = tp.gradient(epoch_loss, params)
    del tp
    optimizer.apply_gradients(zip(gradient, params))
    return epoch_loss

for epoch in range(epochs):
    epoch_loss = gradient_descent()
    print(epoch, epoch_loss.numpy())

####### The remainder of the code is identical to advection.py
#
# Output some informative information 
print("After training, the losses are:")
print("Initial = ", loss_dirichlet(T_initial, X_initial, vals_initial))
print("Dirichlet = ", loss_dirichlet(T_dirichlet, X_dirichlet, vals_dirichlet))
print("Neumann = ", loss_neumann(T_neumann, X_neumann, flux_neumann))
print("DE = ", loss_de(T_interior, X_interior))

# Display the results graphically
fig = plt.figure()
axis = plt.axes(xlim = (-1, 1), ylim = (-0.1, 1.1))
line, = axis.plot([], [], linewidth = 2)
stuff_to_animate = [axis.plot([], [], linewidth = 2, color = 'k', label = 'Analytic')[0], axis.plot([], [], linewidth = 2, color = 'r', linestyle = '--', label = 'PINN')[0], axis.annotate(0, xy = (-0.9, 0.9), xytext = (-0.9, 0.9), fontsize = 13)]
def init():
    stuff_to_animate[0].set_data([], [])
    stuff_to_animate[1].set_data([], [])
    stuff_to_animate[2].set_text("t = 0.00")
    return stuff_to_animate
xdata = tf.constant(np.linspace(-1, 1, 1000), dtype = tf.float32)
def animate(i):
    t = 0.01 * i
    tdata = tf.constant([t] * len(xdata), dtype = tf.float32)
    stuff_to_animate[0].set_data(xdata, front(xdata - velocity * t))
    stuff_to_animate[1].set_data(xdata, model(tf.stack([tdata, xdata], 1)))
    stuff_to_animate[2].set_text("t = " + str(round(t, 2)))
    return stuff_to_animate
anim = ani.FuncAnimation(fig, animate, init_func = init, frames = 101, interval = 20, blit = True)
plt.grid()
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("Analytic and PINN solutions")
anim.save('advection_descent.gif', fps = 30)
plt.show()
sys.exit(0)
