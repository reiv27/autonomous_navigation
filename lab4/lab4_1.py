import timeit
import numpy as np
import matplotlib.pyplot as plt

from sampling_methods import *


N = 10000
b = 1
var = 1

time1 = timeit.timeit(lambda: [sample_normal_distribution(b) for _ in range(N)], number=1)
print(f"sample_normal_distribution: {time1:.4f} s")

time2 = timeit.timeit(lambda: [sample_distribution(f, b) for _ in range(N)], number=1)
print(f"sample_distribution: {time2:.4f} s")

time3 = timeit.timeit(lambda: [sample_normal_box_muller(0.0, var) for _ in range(N)], number=1)
print(f"sample_normal_box_muller: {time3:.4f} s")

time_numpy = timeit.timeit(lambda: np.random.normal(0.0, np.sqrt(var), N), number=1)
print(f"numpy.random.normal: {time_numpy:.4f} s", end="\n\n")

samples_1 = np.array([sample_normal_distribution(b) for _ in range(N)])
samples_2 = np.array([sample_distribution(f, b) for _ in range(N)])
samples_3 = np.array([sample_normal_box_muller(0.0, var) for _ in range(N)])
samples_numpy = np.random.normal(0.0, np.sqrt(var), N)

print(f"sample_normal_distribution: mean={np.mean(samples_1):.4f}, std={np.std(samples_1):.4f}")
print(f"sample_distribution: mean={np.mean(samples_2):.4f}, std={np.std(samples_2):.4f}")
print(f"sample_normal_box_muller: mean={np.mean(samples_3):.4f}, std={np.std(samples_3):.4f}")
print(f"numpy.random.normal: mean={np.mean(samples_numpy):.4f}, std={np.std(samples_numpy):.4f}")

plt.figure(figsize=(8, 8))
plt.hist(samples_1, bins=100, alpha=0.5, label='sample_normal_distribution')
plt.hist(samples_2, bins=100, alpha=0.5, label='sample_distribution')
plt.hist(samples_3, bins=100, alpha=0.5, label='sample_normal_box_muller')
plt.hist(samples_numpy, bins=100, alpha=0.5, label='numpy.random.normal')
plt.legend()
plt.show()
