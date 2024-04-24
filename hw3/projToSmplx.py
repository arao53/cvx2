import numpy as np

# Project to Simplex Function

#  Input: v is n x 1 column matrix
#  Output: x is the result of projection
#  Description: project v into simplex

def find(u, sv):
    condition = u > ( (sv-1) /  range(1,len(u)+1) )
    for i in reversed(range(len(condition))): 
        if condition[i]:
            return i
        
def projToSmplx(v):
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    ind = find(u,sv)
    theta = (sv[ind]-1) / (ind+1)
    x = np.maximum(v - theta, 0)
    return x