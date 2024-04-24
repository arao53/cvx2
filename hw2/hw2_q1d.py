import numpy as np
import matplotlib.pyplot as plt

# setup
n = 1000
m = 200 
lam = 0.01
k = 5
loc = 0
scale = np.sqrt(1/m)

np.random.seed(0)

# generate data
A = np.random.normal(loc, scale, size = (m,n))
x = np.zeros(n)
x[:k] = np.ones(k)
b = np.dot(A,x)

# define subgradient
def subgrad(x):
    return np.dot(A.T, np.dot(A,x) - b) + lam*np.sign(x)

def get_obj(x):
    return 0.5*np.linalg.norm(np.dot(A,x) - b)**2 + lam*np.linalg.norm(x,1)

# define subgradient parameters
steps = 500
idx = range(steps-1)

obj_d = np.ones(steps) * np.nan # 1/sqrt(k) step size
obj_e = np.ones(steps) * np.nan # polyak step length

# part A - constant step size
obj_a = np.ones(steps) * np.nan 
x_a = np.zeros((n, steps)) * np.nan
alpha = 0.005
beta = 0.99
x_a[:,0] = np.zeros(n)
obj_a[0] = get_obj(x_a[:, 0])

for i in idx:   # run subgradient descent
        # calculate step
    if i > 0:
        ball = beta * (x_a[:, i] - x_a[:, i-1])
    else: 
        ball = np.zeros(n)
    x_a[:, i+1] = x_a[:, i] - alpha*subgrad(x_a[:, i]) + ball
        # update objective
    obj_a[i+1] = get_obj(x_a[:, i+1])


# part B - constant step length
obj_b = np.ones(steps) * np.nan 
x_b = np.zeros((n, steps)) * np.nan
x_b[:,0] = np.zeros(n)
gamma = 0.01
obj_b[0] = get_obj(x_b[:, 0])

for i in idx:   # run subgradient descent
        # calculate step
    g = subgrad(x_b[:, i])
    a = gamma / np.linalg.norm(g,2)
    if i > 0:
        ball = beta * (x_b[:, i] - x_b[:, i-1])
    else:
        ball = np.zeros(n)
    x_b[:, i+1] = x_b[:, i] - a*g + ball
        # update objective
    obj_b[i+1] = get_obj(x_b[:, i+1])


# part C - 1/sqrt(k) step size
obj_c = np.ones(steps) * np.nan 
x_c = np.zeros((n, steps)) * np.nan
x_c[:,0] = np.zeros(n)
obj_c[0] = get_obj(x_c[:, 0])

for i in idx:   # run subgradient descent
        # calculate step
    a = 1/np.sqrt(i+1)
    if i > 0:
        ball = beta * (x_c[:, i] - x_c[:, i-1])
    else:
        ball = np.zeros(n)
    x_c[:, i+1] = x_c[:, i] - a*subgrad(x_c[:, i]) + ball
        # update objective
    obj_c[i+1] = get_obj(x_c[:, i+1])

# part D - 1/k step size
obj_d = np.ones(steps) * np.nan 
x_d = np.zeros((n, steps)) * np.nan
x_d[:,0] = np.zeros(n)
obj_d[0] = get_obj(x_d[:, 0])

for i in idx:   # run subgradient descent
        # calculate step
    a = 1/(i+1)
    if i > 0:
        ball = beta * (x_d[:, i] - x_d[:, i-1])
    else:
        ball = np.zeros(n)
    x_d[:, i+1] = x_d[:, i] - a*subgrad(x_d[:, i]) + ball
        # update objective
    obj_d[i+1] = get_obj(x_d[:, i+1])


# # part E - Polyak step length
obj_e = np.ones(steps) * np.nan
x_e = np.zeros((n, steps)) * np.nan
x_e[:,0] = np.zeros(n)
obj_e[0] = get_obj(x_e[:, 0])
f_star = np.min([obj_a[-1], obj_b[-1], obj_c[-1], obj_d[-1]])

for i in idx:   # run subgradient descent
        # calculate step
    g = subgrad(x_e[:, i])
    a = (obj_e[i] - f_star) / np.linalg.norm(g,2)**2   # assume optimal function value is best known solution
    if i > 0:
        ball = beta * (x_b[:, i] - x_b[:, i-1])
    else:
        ball = np.zeros(n)
    x_e[:, i+1] = x_e[:, i] - a*g + ball
        # update objective
    obj_e[i+1] = get_obj(x_e[:, i+1])

# # plot
fig, ax = plt.subplots()
ax.plot(obj_a, alpha = 0.75, label = "Step Size = {}".format(alpha))
ax.plot(obj_b, alpha = 0.75, label = "Step Length = {}".format(gamma))
ax.plot(obj_c, alpha = 0.75, label = "Step Size = 1/sqrt(k)")
ax.plot(obj_d, alpha = 0.75, label = "Step Size = 1/k")
ax.plot(obj_e, alpha = 0.75, label = "Polyak Step Length")
ax.set(xlabel='Iteration', 
       ylabel='Objective Value',
       title='Subgradient Method Convergence (Beta = {})'.format(beta))
ax.set_ylim([0,3])
ax.set_xlim([0,steps])
ax.legend()

fig.savefig("hw2_q1d3.png")