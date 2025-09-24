import numpy as np
import matplotlib.pyplot as plt


def prob(x, var, mu=0.0):
  sigma = np.sqrt(var)
  return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def landmark_detection_model(z, x, m):
  '''
  Landmark Detection Model
  Args:
    z: measurement [d, var]
    x: state [x, y]
    m: landmark [x_m, y_m]
  Returns:
    p_det: probability of detection
  '''
  d_hat = np.sqrt((m[0] - x[0])**2 + (m[1] - x[1])**2)
  p_det = prob(d_hat - z[0], z[1])
  return p_det


x0 = np.array([
  [10.0],
  [8.0],
])
m0 = np.array([
  [12.0],
  [4.0],
])
z0 = np.array([
  [3.9],
  [1.0],
])

x1 = np.array([
  [6.0],
  [3.0],
])
m1 = np.array([
  [5.0],
  [7.0],
])
z1 = np.array([
  [3.9],
  [1.5],
])

p00 = landmark_detection_model(z0.reshape(1, 2)[0], x0.reshape(1, 2)[0], m0.reshape(1, 2)[0])
p01 = landmark_detection_model(z1.reshape(1, 2)[0], x0.reshape(1, 2)[0], m1.reshape(1, 2)[0])
print(f"p00: {p00}, p01: {p01}")
print(p00 * p01)
p10 = landmark_detection_model(z0.reshape(1, 2)[0], x1.reshape(1, 2)[0], m0.reshape(1, 2)[0])
p11 = landmark_detection_model(z1.reshape(1, 2)[0], x1.reshape(1, 2)[0], m1.reshape(1, 2)[0])
print(f"p10: {p10}, p11: {p11}")
print(p10 * p11)

num_points = 500
start = x0.flatten()
end = x1.flatten()
strip_width = 10.0
strip_points = 100
direction = end - start
direction_norm = direction / np.linalg.norm(direction)
perp = np.array([-direction_norm[1], direction_norm[0]])
center_points = np.linspace(start, end, num_points)
width_offsets = np.linspace(-strip_width/2, strip_width/2, strip_points)
points = []
for center in center_points:
  for offset in width_offsets:
    pt = center + perp * offset
    points.append(pt)
points = np.array(points)
probabilities = []
for pt in points:
  p0 = landmark_detection_model(z0.reshape(1, 2)[0], pt, m0.reshape(1, 2)[0])
  p1 = landmark_detection_model(z1.reshape(1, 2)[0], pt, m1.reshape(1, 2)[0])
  probabilities.append(p0 * p1)
probabilities = np.array(probabilities)

X = points[:, 0].reshape(num_points, strip_points)
Y = points[:, 1].reshape(num_points, strip_points)
Z = probabilities.reshape(num_points, strip_points)

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, Z, color='green', alpha=0.5, rstride=10, cstride=2)

ax.scatter(x0[0], x0[1], 0, s=300, color='blue', marker='*', label='Sirius')
ax.scatter(x1[0], x1[1], 0, s=300, color='black', marker='s', label='Home')
ax.scatter(m0[0], m0[1], 0, s=300, color='red', marker='h', label='Marker 0')
ax.scatter(m1[0], m1[1], 0, s=300, color='yellow', marker='X', label='Marker 1')

ax.set_xlabel('x, m')
ax.set_ylabel('y, m')
ax.set_zlabel('Probability')
ax.set_title('Landmark Detection Model')
ax.legend()
plt.show()