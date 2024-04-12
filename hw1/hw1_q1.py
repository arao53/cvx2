from time import time, sleep
import dask
import dask.delayed
import numpy as np
import matplotlib.pyplot as plt


def inprod(x, y):
    return np.dot(x, y)

def argmax(x):
    return np.argmax(x)


n_trials = 100
n, m = int(2e7), int(2**2)

data = np.random.randn(m, n)

serial_times = []
dask_times = []
numpy_times = []

for i in range(n_trials):
    x = np.random.randn(n)

    start = time()
    output = []

    # Part (b): serial computation
    for j in range(data.shape[0]):
        tmp = inprod(data[j], x)
        if j == 0:
            max_val = tmp
            subgrad = data[j]
        else:
            if tmp > max_val:
                max_val = tmp
                subgrad = data[j]
    serial_times.append(time() - start)


    start = time()
    output_c = []
    # Part (c): using dask.delayed()
    for k in range(data.shape[0]):
        output_c.append(dask.delayed(inprod)(data[k], x))
    idx = dask.delayed(argmax)(output_c)
    subgrad_c = data[idx.compute()]

    dask_times.append(time() - start)

    # ===

    # Part (d): using numpy.matmul()
    start = time()
    output = np.matmul(data, x)
    idx = np.argmax(output)
    subgrad_d = data[idx]
    numpy_times.append(time() - start)


# part(e): plot histogram of times

fig, ax = plt.subplots()
ax.hist(serial_times, bins=20, alpha=0.5, label='Serial')
ax.hist(dask_times, bins=20, alpha=0.5, label='Dask')
ax.hist(numpy_times, bins=20, alpha=0.5, label='Numpy')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency')
plt.legend()
fig.savefig('hw1_q1.png')

# ===

# part(f): visualize Dask computation graph.
# subgrad_c.visualize()


# ===
# n, m = 5,4 
# A = np.random.randn(m, n)
# x = np.random.randn(n)
# for k in range(A.shape[0]):
#     output_c.append(dask.delayed(inprod)(A[k], x))
# idx = dask.delayed(argmax)(output_c)
# idx.visualize()
