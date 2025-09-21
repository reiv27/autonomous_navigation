import numpy as np
import matplotlib.pyplot as plt


def diffdrive(x, y, theta, v_l, v_r, t, l):
  if abs(v_r - v_l) < 1e-8:
    R = 0
  else:
    R = l / 2 * (v_r + v_l) / (v_r - v_l)
  w = (v_r - v_l) / l

  r = np.array([
    [x],
    [y],
    [theta],
  ])

  dt = 0.01
  N = int(t / dt)
  for _ in range(N):
    if abs(v_r - v_l) < 1e-6:
      v = (v_r + v_l) / 2
      r = np.array([
        [r[0,0] + v * np.cos(r[2,0]) * dt],
        [r[1,0] + v * np.sin(r[2,0]) * dt],
        [theta],
      ])
    else:
      iic = np.array([
        [r[0,0] - R * np.sin(r[2,0])],
        [r[1,0] + R * np.cos(r[2,0])],
      ])

      r = np.array([
        [np.cos(w*dt), -np.sin(w*dt), 0],
        [np.sin(w*dt),  np.cos(w*dt), 0],
        [           0,            0,  1],
      ]) @ np.array([
        [r[0,0] - iic[0,0]],
        [r[1,0] - iic[1,0]],
        [           r[2,0]],
      ]) + np.array([
        [iic[0,0]],
        [iic[1,0]],
        [w*dt],
      ])
  
  x_n = float(r[0,0])
  y_n = float(r[1,0])
  theta_n = float(r[2,0])

  return x_n, y_n, theta_n

x_init = 1.5
y_init = 2.0
theta_init = np.pi/2
l = 0.5

# c_i = (v_l = ..., v_r = ..., t = ...)
c_1 = [0.3, 0.3, 3]
c_2 = [0.1, -0.1, 1]
c_3 = [0.2, 0, 2]

x_points = [x_init]
y_points = [y_init]
theta_points = [theta_init]

x_1, y_1, theta_1 = diffdrive(x_init, y_init, theta_init, c_1[0], c_1[1], c_1[2], l)
x_points.append(x_1)
y_points.append(y_1)
theta_points.append(theta_1)

x_2, y_2, theta_2 = diffdrive(x_1, y_1, theta_1, c_2[0], c_2[1], c_2[2], l)
x_points.append(x_2)
y_points.append(y_2)
theta_points.append(theta_2)

x_3, y_3, theta_3 = diffdrive(x_2, y_2, theta_2, c_3[0], c_3[1], c_3[2], l)
x_points.append(x_3)
y_points.append(y_3)
theta_points.append(theta_3)

# print(x_points)
# print(y_points)
# print(theta_points)

plt.figure(figsize=(8, 8))
plt.plot(x_points, y_points, 'r--', label='Robot Path')
plt.scatter(x_points[0], y_points[0], s=50, color='blue', label='Start')
plt.scatter(x_points[-1], y_points[-1], s=50, color='black', label='End')

arrow_length = 0.1
for x, y, theta in zip(x_points, y_points, theta_points):
    dx = arrow_length * np.cos(theta)
    dy = arrow_length * np.sin(theta)
    plt.arrow(x, y, dx, dy, head_width=0.03, head_length=0.04, fc='g', ec='g')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Robot Path and Orientation')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()