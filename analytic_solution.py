import os
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as ani

velocity = 1.2

def solution(t, x):
    window = 0.2
    return np.where(x - velocity * t <= -1, 1, np.where(x - velocity * t >= -1 + 2 * window, 0, 0.5 + (x - velocity * t + 1 - window) * (np.power(x - velocity * t + 1 - window, 2) - 3 * np.power(window, 2)) / 4 / np.power(window, 3)))

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
t = np.linspace(0, 1, 100)
x = np.linspace(-1, 1, 100)
X, T = np.meshgrid(x, t)
u = np.array(solution(np.ravel(T), np.ravel(X)))
U = u.reshape(X.shape)
ax.plot_surface(X, T, U)
ax.view_init(elev = 34, azim = -74)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u")
plt.title("Analytic solution")
plt.savefig("analytic_solution.png")
plt.show()
plt.close()

fig = plt.figure()
axis = plt.axes(xlim = (-1, 1), ylim = (-0.1, 1.1))
line, = axis.plot([], [], linewidth = 2, color = 'r')
time = axis.annotate(0, xy = (-0.9, 0.9), xytext = (-0.9, 0.9), fontsize = 13)
def init():
    line.set_data([], [])
    time.set_text("t = 0.00")
    return time, line,
def animate(i):
    t = 0.01 * i
    xdata = np.linspace(-1, 1, 1000)
    ydata = solution(t, xdata)
    line.set_data(xdata, ydata)
    time.set_text("t = " + str(round(t, 2)))
    return time, line,
anim = ani.FuncAnimation(fig, animate, init_func = init, frames = 101, interval = 20, blit = True)
plt.grid()
plt.xlabel("x")
plt.ylabel("u")
plt.title("Analytic solution")
anim.save('analytic_solution.gif', fps = 30)
plt.show()


