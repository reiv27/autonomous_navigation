import numpy as np


# 1
def sample_normal_distribution(b):
    samples = np.random.uniform(-b, b, 12)
    return 0.5 * sum(samples)


# 2
def sample_distribution(f, b):
    x_array = np.linspace(-b, b, 10000)
    f_max = max(f(x_array))
    while True:
        x = np.random.uniform(-b, b)
        y = np.random.uniform(0, f_max)
        if y <= f(x):
            return x


def f(x, mu=0.0, var=0.0001):
    sigma = np.sqrt(var)
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# 3
def sample_normal_box_muller(mu=0.0, var=1.0):
    u1 = np.random.rand()
    u2 = np.random.rand()
    z = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
    return mu + np.sqrt(var) * z