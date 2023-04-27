import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
tf.keras.utils.set_random_seed(12)

#########################################################################################
# This code trains a neural network so that its derivative is a specified function
# More precisely, the neural network has 1 input, x, and 1 output, u = u(x)
# It is trained so that du/dx = fcn
# This is different than a usual nonlinear regression neural network, where the values,
# u, are known at certain x points, and the NN is trained to produce these values.
# This code utilizes TensorFlow's automatic differentiation
# However, the implementation deliberately tries to use an approach that is as simple
# as possible.  For instance, the neural network descent process does not utilize the
# automatic derivatives of the neural network with respect to its internal parameters
# in a custom gradient descent algorithm: rather, the bulk-standard SGD is used.
#########################################################################################

X = tf.constant(np.linspace(0, 2 * np.pi, 1000), dtype = tf.float32) # domain of the problem
fcn = tf.math.cos(X)                                                 # function to be integrated
value_at_0 = 0.7                                                     # constant of integration (value at X=0)
true_integral = value_at_0 + tf.math.sin(X)                          # for comparing the results
# Another example:
#fcn = tf.math.sin(X)                                                 # function to be integrated
#value_at_0 = 0.0                                                     # constant of integration (value at X=0)
#true_integral = value_at_0 - (tf.math.cos(X) - 1)                    # for comparing the results
# Another example:
#fcn = tf.math.exp(0.3 * X)                                           # function to be integrated
#value_at_0 = 2.0                                                     # constant of integration (value at X=0)
#true_integral = value_at_0 + (tf.math.exp(0.3 * X) - 1) / 0.3        # for comparing the results
# Another example:
#fcn = tf.math.pow(X + 0.5, -1)                                       # function to be integrated
#value_at_0 = 1.0                                                     # constant of integration
#true_integral = value_at_0 + tf.math.log(X + 0.5) - tf.math.log(0.5) # for comparing the results

#########################################################################################
# The code works by minimising the loss function, which is a weighted sum of:
# - loss_bdy, which attempts to ensure that u(X[0]) - value_at_zero = 0, and
# - loss_de, which attempts to ensure that du/dx - fcn = 0 at all the points defined by X
# The relative weighting of each term obviously impacts the outcome.
#########################################################################################

def loss_bdy(x, val_at_zero):
    ''' Evaluate the boundary condition, ie u(0) - val_at_zero, where u is given by the NN model
    '''
    val = model(tf.convert_to_tensor([x[0]], dtype = tf.float32)) - val_at_zero # "model" is the NN predicted value, given x (see below)
    return tf.reduce_mean(tf.square(val))

def loss_de(x, function_values):
    ''' Returns sum_over_x(|du/dx - function_values|^2) / number_of_x_points
    '''
    # First, use TensorFlow automatic differentiation to evaluate du/dx, at the points x, where u is given by the NN model
    with tf.GradientTape(persistent = True) as tp:
        tp.watch(x)
        u = model(x) # "model" is the NN predicted value, given x (see below)
    u_x = tp.gradient(u, x)
    del tp
    # The loss is just the mean-squared du/dx - function_values
    return tf.reduce_mean(tf.square(u_x - function_values))

@tf.function # decorate for speed
def loss(ytrue, ypred):
    ''' The loss used by the training algorithm.  Note that ytrue and ypred are not used,
    but TensorFlow specifies these arguments
    '''
    bdy_weight = 1
    de_weight = 1
    return bdy_weight * loss_bdy(X, value_at_0) + de_weight * loss_de(X, fcn)

###############################################################################################
# The remainder is just usual neural-net stuff with Keras
# Note, the goodness of the fit depends quite strongly on the DE (the functional form of fcn),
# the constant of integration, the architecture of the NN, and other hyperparameters
###############################################################################################
depth = 5          # depth of NN
width = 10         # width of fully-connected NN
activation = 'elu' # alternatives 'selu', 'softplus', 'sigmoid', 'tanh', 'elu', 'relu'
epochs = 2000      # training epochs
batch_size = 1000  # batch size

# Create and compile the Neural Net
model = Sequential()
model.add(Dense(width, kernel_initializer = 'normal', activation = activation, input_shape = (1, )))
for d in range(1, depth):
    model.add(Dense(width, kernel_initializer = 'normal', activation = activation))
model.add(Dense(1, kernel_initializer = 'normal', activation = activation))
model.compile(loss = loss, optimizer = 'adam') # note the specification of the loss function

history = model.fit(X, fcn, epochs = epochs, batch_size = batch_size, verbose = 1) # fit: note that X and fcn are actually not used because the loss function uses different arguments

# plot results
plt.figure()
plt.plot(X, fcn, 'g-', alpha = 0.5, label = 'Function')
plt.scatter([0], [value_at_0], s=20, c='k', label = 'Specified value at $x=0$')
plt.plot(X, true_integral, 'k-', label = 'Exact integral')
plt.plot(X, model.predict(X), 'r--', label = 'NN approximation')
plt.legend()
plt.grid()
plt.xlabel("x")
plt.title("Integral of a function")
plt.savefig("integrate.png", bbox_inches = 'tight')
plt.show()

sys.exit(0)
