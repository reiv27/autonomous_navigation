import numpy as np
import matplotlib.pyplot as plt
from sampling_methods import sample_normal_distribution


def sample_motion_model(x, u, alpha):
  '''
  Sample Odometry Motion Model
  Args:
    x: current state [x, y, theta]
    u: odometry [d_rot1, d_rot2, d_trans]
    alpha: noise model parameters [alpha1, alpha2, alpha3, alpha4]
  Returns:
    x_new: new state [x_new, y_new, theta_new]
  '''
  d_rot1_est = u[0] + sample_normal_distribution(alpha[0] * abs(u[0]) + alpha[1] * u[2])
  d_rot2_est = u[1] + sample_normal_distribution(alpha[0] * abs(u[1]) + alpha[1] * u[2])
  d_trans_est = u[2] + sample_normal_distribution(alpha[2] * u[2] + alpha[3] * (abs(u[0]) + abs(u[1])))
  x_new = x[0] + d_trans_est * np.cos(x[2] + d_rot1_est)
  y_new = x[1] + d_trans_est * np.sin(x[2] + d_rot1_est)
  theta_new = x[2] + d_rot1_est + d_rot2_est
  return np.array([x_new, y_new, theta_new])


x_init = np.array([
  [2.0],
  [4.0],
  [0.0],
])

u = np.array([
  [np.pi/2],
  [    0.0],
  [    1.0],
])

alpha = np.array([
  [ 0.1],
  [ 0.1],
  [0.01],
  [0.01],
])

N = 5000
x_points = [x_init[0,0]]
y_points = [x_init[1,0]]
theta_points = [x_init[2,0]]

for i in range(N):
  x_new = sample_motion_model(x_init.reshape(1,3)[0], u.reshape(1,3)[0], alpha.reshape(1,4)[0])
  x_points.append(x_new[0])
  y_points.append(x_new[1])
  theta_points.append(x_new[2])

plt.figure(figsize=(16, 16))
plt.scatter(x_points[0], y_points[0], s=300, color='green', marker='*', label='Start')
plt.scatter(x_points[1:], y_points[1:], s=20, color='blue', label='Estimated States')
# arrow_scale = 0.1
# for x, y, theta in zip(x_points[1:], y_points[1:], theta_points[1:]):
#     plt.arrow(
#         x, y,
#         arrow_scale * np.cos(theta),
#         arrow_scale * np.sin(theta),
#         head_width=0.01, head_length=0.05, fc='red', ec='red', length_includes_head=True
#     )
plt.xlabel(r'$x, m$')
plt.ylabel(r'$y, m$')
plt.title(r'Robot Path and Orientation')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()