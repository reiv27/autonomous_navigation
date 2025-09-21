import math
import numpy as np
import matplotlib.pyplot as plt


scan = np.loadtxt('laserscan.dat')
angle = np.linspace(-math.pi/2, math.pi/2, np.shape(scan)[0], endpoint='true')

x = 1.0
y = 0.5
theta = np.pi/4
lx = 0.2
ly = 0.0
ltheta = np.pi

T01 = np.array([
  [np.cos(theta), -np.sin(theta), x],
  [np.sin(theta),  np.cos(theta), y],
  [            0,              0, 1]
])
T12 = np.array([
  [np.cos(ltheta), -np.sin(ltheta), lx],
  [np.sin(ltheta),  np.cos(ltheta), ly],
  [             0,               0,  1]
])
T02 = T01 @ T12

x_points = []
y_points = []
for i in range(scan.shape[0]):
  T23 = np.array([
    [1, 0, scan[i] * np.cos(angle[i])],
    [0, 1, scan[i] * np.sin(angle[i])],
    [0, 0,                          1]
  ])
  T03 = T02 @ T23
  x_points.append(T03[0, 2])
  y_points.append(T03[1, 2])

plt.scatter(x, y, color='black', label='Robot')
plt.scatter(T02[0, 2], T02[1, 2], color='blue', label='LiDAR')
plt.scatter(x_points, y_points, s=1, color='red', label='Point Cloud')
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Laser Scan')
plt.xlabel(r'$x, m$')
plt.ylabel(r'$y, m$')
plt.grid(True)
plt.legend()
plt.show()  