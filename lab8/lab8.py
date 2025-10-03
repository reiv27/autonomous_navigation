import numpy as np
import matplotlib.pyplot as plt

res = 10                 # grid resolution
map_len = 200                   # map length
robot_pos = 0.0                 # robot position (at c0)
measurements = np.array([101, 82, 91, 112, 99, 151, 96, 85, 99, 105], dtype=float)

p_free = 0.3                  # p for cells closer than z
p_occ  = 0.6                  # p for cells in [z, z+20]
extra = 20.0                # width of "hit" interval
prior = 0.5

def logit(p: np.ndarray):
    return np.log(p / (1.0 - p))

def sigmoid(x: np.ndarray):
    return 1.0 / (1.0 + np.exp(-x))

c = np.arange(0, map_len, res, dtype=float)  # cell coordinates (cm)

near = np.arange(0, map_len, res, dtype=float)
far  = near + res
centers = near + res * 0.5

l = np.zeros(c.shape[0], dtype=float)

for z in measurements:
    p_m = np.full(c.shape[0], 0.5, dtype=float)

    # p_m[c < z] = p_free
    p_m[far <= z] = p_free

    start = z
    end = z + extra
    mask_hit = (c >= start) & (c <= end)
    p_m[mask_hit] = p_occ

    l += logit(p_m)

m = sigmoid(l)

plt.figure(figsize=(12, 8))
plt.plot(c, m)
plt.xlabel("cells, cm")
plt.ylabel("p_occ")
plt.grid(True)
plt.show()