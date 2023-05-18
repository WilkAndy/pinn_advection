##################################################################################
# Creates observations.csv, which contains (synthetic) observations of "u"
# for the advection problem
#
# The advection equation is
# du/dt = - velocity * du/dx
# The initial condition is
#                | 1     if x <= -1
# u = front(x) = | cubic if -1 < x < 1 + 2 * window
#                | 0     if x >= 1 + 2 * window
# where the cubic is chosen so u is differentiable.  As "window" tends to zero
# this becomes a step function.
# The boundary conditions are u(x=-1) = 1, and du/dx(x=1) = 0
# The solution of the advection equation is u = front(x - velocity * t)
##################################################################################

import os
import sys
import tensorflow as tf

# note: velocity cannot be too large or too small, otherwise boundary conditions
# will be violated
velocity = 1.2 

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

num_points = 10000
tmax = 1
X = tf.random.uniform(shape = [num_points], minval = -1, maxval = 1, dtype = tf.float32).numpy()
T = tf.random.uniform(shape = [num_points], minval = 0, maxval = tmax, dtype = tf.float32).numpy()
u = front(X - velocity * T).numpy()
with open("observations.csv", "w") as f:
    f.write("T,X,u\n")
    for pt in range(num_points):
        f.write(str(T[pt]) + "," + str(X[pt]) + "," + str(u[pt]) + "\n")

sys.exit(0)
