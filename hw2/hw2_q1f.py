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

# define lagrange functions
def subgrad(x):
    return np.dot(A.T, np.dot(A,x) - b) + lam*np.sign(x)

def dLx(x,y,v, A, b, rho):
    diff = x - y
    pos_diff = np.max([diff, np.zeros_like(diff)], axis = 0)
    return np.dot(A.T, np.dot(A,x) - b) + v + rho*(pos_diff)

def dLy(x,y,v, lam, rho):
    diff = x - y
    pos_diff = np.max([diff, np.zeros_like(diff)], axis = 0)
    return lam * np.sign(y) - v - rho*(pos_diff)

def dLv(x,y):
    diff = x - y
    pos_diff = np.max([diff, np.zeros_like(diff)], axis = 0)
    return np.linalg.norm(pos_diff,2)

def get_obj(x):
    return 0.5*np.linalg.norm(np.dot(A,x) - b)**2 + lam*np.linalg.norm(x,1)

# define subgradient parameters
steps = 500
idx = range(steps-1)

# part A - constant step size
obj_a = np.ones(steps) * np.nan 
x_a = np.zeros((n, steps)) * np.nan
y_a = np.zeros((n, steps)) * np.nan
v_a = np.zeros(steps) * np.nan
alpha = 0.005
rho = 1e-8

# set initial guess
x_a[:,0] = np.zeros(n)
y_a[:,0] = np.zeros(n)
v_a[0] = 0

obj_a[0] = get_obj(x_a[:, 0])

for i in idx:   # run primal dual method
        # calculate step
    x_a[:, i+1] = x_a[:, i] - alpha*dLx(x_a[:, i], y_a[:, i], v_a[i], A, b, rho)
    y_a[:, i+1] = y_a[:, i] - alpha*dLy(x_a[:, i], y_a[:, i], v_a[i], lam, rho)
    v_a[i+1] = v_a[i] + rho*dLv(x_a[:, i], y_a[:, i])
    
        # update objective
    obj_a[i+1] = get_obj(x_a[:, i+1])

# part B - constant step length
obj_b = np.ones(steps) * np.nan 
x_b = np.zeros((n, steps)) * np.nan
y_b = np.zeros((n, steps)) * np.nan
v_b = np.zeros(steps) * np.nan
gamma = 0.01    # step length

# set initial guess
x_b[:,0] = np.zeros(n)
y_b[:,0] = np.zeros(n)
v_b[0] = 0
obj_b[0] = get_obj(x_b[:, 0])

for i in idx:   
        # calculate step
    g = subgrad(x_b[:, i])
    a = gamma / np.linalg.norm(g,2)
    
    x_b[:, i+1] = x_b[:, i] - alpha*dLx(x_b[:, i], y_b[:, i], v_b[i], A, b, rho)
    y_b[:, i+1] = y_b[:, i] - alpha*dLy(x_b[:, i], y_b[:, i], v_b[i], lam, rho)
    v_b[i+1] = v_b[i] + rho*dLv(x_b[:, i], y_b[:, i])
    
        # update objective
    obj_b[i+1] = get_obj(x_b[:, i+1])


# part C - 1/sqrt(k) step size
obj_c = np.ones(steps) * np.nan
x_c = np.zeros((n, steps)) * np.nan
y_c = np.zeros((n, steps)) * np.nan
v_c = np.zeros(steps) * np.nan
x_c[:,0] = np.zeros(n)
y_c[:,0] = np.zeros(n)
v_c[0] = 0
obj_c[0] = get_obj(x_c[:, 0])

for i in idx:   # run primal dual method
        # calculate step
    a = 1/np.sqrt(i+1)
    
    x_c[:, i+1] = x_c[:, i] - alpha*dLx(x_c[:, i], y_c[:, i], v_c[i], A, b, rho)
    y_c[:, i+1] = y_c[:, i] - alpha*dLy(x_c[:, i], y_c[:, i], v_c[i], lam, rho)
    v_c[i+1] = v_c[i] + rho*dLv(x_c[:, i], y_c[:, i])
    
        # update objective
    obj_c[i+1] = get_obj(x_c[:, i+1])


# part D - 1/k step size
obj_d = np.ones(steps) * np.nan
x_d = np.zeros((n, steps)) * np.nan
y_d = np.zeros((n, steps)) * np.nan
v_d = np.zeros(steps) * np.nan
x_d[:,0] = np.zeros(n)
y_d[:,0] = np.zeros(n)
v_d[0] = 0

obj_d[0] = get_obj(x_d[:, 0])

for i in idx:   # run primal dual method
        # calculate step
    a = 1/(i+1)
    
    x_d[:, i+1] = x_d[:, i] - alpha*dLx(x_d[:, i], y_d[:, i], v_d[i], A, b, rho)
    y_d[:, i+1] = y_d[:, i] - alpha*dLy(x_d[:, i], y_d[:, i], v_d[i], lam, rho)
    v_d[i+1] = v_d[i] + rho*dLv(x_d[:, i], y_d[:, i])
    
        # update objective
    obj_d[i+1] = get_obj(x_d[:, i+1])

# part E - Polyak step length
obj_e = np.ones(steps) * np.nan
x_e = np.zeros((n, steps)) * np.nan
y_e = np.zeros((n, steps)) * np.nan
v_e = np.zeros(steps) * np.nan
x_e[:,0] = np.zeros(n)
y_e[:,0] = np.zeros(n)
v_e[0] = 0
obj_e[0] = get_obj(x_e[:, 0])
f_star = get_obj(x)

for i in idx:   # run primal dual method
        # calculate step
    a = (obj_e[i] - f_star) / np.linalg.norm(subgrad(x_e[:, i]),2)**2
    
    x_e[:, i+1] = x_e[:, i] - a*dLx(x_e[:, i], y_e[:, i], v_e[i], A, b, rho)
    y_e[:, i+1] = y_e[:, i] - a*dLy(x_e[:, i], y_e[:, i], v_e[i], lam, rho)
    v_e[i+1] = v_e[i] + rho*dLv(x_e[:, i], y_e[:, i])
    
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
       title='Primal-Dual Method (Rho = {})'.format(rho))
ax.set_ylim([0,3])
ax.set_xlim([0,steps])
ax.legend()

fig.savefig("hw2_q1f_3.png")