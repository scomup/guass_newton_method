import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from math_tools import *

# We try to optimize an exponential objective function([Magnusson 2009] 6.9), using newton's method.
# x = (T*dT*a - b) # The transformation error (vector3)
# T(p) # The parameters(6Dof) of transformation matrix(T), We combine the 3D translation vector and the lie algebra so(3) as transformation parameters.


d1 = 1.
d2 = 0.01
cov = np.eye(3)
cov_inv = np.linalg.inv(cov)

def x2m(x):
    t = x[0:3]
    R = exp(x[3:6])
    m = np.eye(4)
    m[0:3,0:3] = R
    m[0:3,3] = t
    return m

def trans_error(init_p, a, b):
    T = x2m(init_p)
    return np.dot(T[0:3,0:3],a) + T[0:3,3] - b

def trans_error(init_p, a, b):
    T = x2m(init_p)
    cinv_sqrt = np.linalg.cholesky(cov_inv)
    x_prime = np.dot(T[0:3,0:3],a) + T[0:3,3] - b
    return cinv_sqrt.dot(x_prime)



def fun(init_p, a, b):
    x = trans_error(init_p, a, b)
    f = x.dot(x)
    return f

def calcdxdp(init_p, a):
    cinv_sqrt = np.linalg.cholesky(cov_inv)
    W = cinv_sqrt.T.dot(x2m(init_p)[0:3,:])
    dxdp = np.array([[W[0,0], W[0,1], W[0,2], -W[0,1]*a[2] + W[0,2]*a[1], W[0,0]*a[2] - W[0,2]*a[0], -W[0,0]*a[1] + W[0,1]*a[0]],
                     [W[1,0], W[1,1], W[1,2], -W[1,1]*a[2] + W[1,2]*a[1], W[1,0]*a[2] - W[1,2]*a[0], -W[1,0]*a[1] + W[1,1]*a[0]],
                     [W[2,0], W[2,1], W[2,2], -W[2,1]*a[2] + W[2,2]*a[1], W[2,0]*a[2] - W[2,2]*a[0], -W[2,0]*a[1] + W[2,1]*a[0]]])
    return dxdp



def gradient_hessian(init_p, a, b):
    #f = fun(init_p, a, b)
    e = trans_error(init_p, a, b)
    J = calcdxdp(init_p, a)
    g = J.T.dot(e)
    h = J.T.dot(J)
    return g, h

"""
Generating test data
"""
elements = 100
np.random.seed(seed=0)
A =  (np.random.rand(elements,3)-0.5)*2

B = transform3d(np.array([0.2,0.1,-0.2, 0.3,0.3,0.3]), A.T).T
init_p = np.array([1000,1000,1000, 0.6,0.6,1.2])
B = transform3d(init_p, B.T).T
#Tinit = x2m(init_p)
#T = Tinit.dot(np.eye(4))
fig = plt.figure()
ax = Axes3D(fig)
ax = Axes3D(fig)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

cur_A = A
while(True):
    plt.cla()
    Ashow = transform3d(init_p, cur_A.T).T
    ax.scatter3D(Ashow[:,0],Ashow[:,1],Ashow[:,2], c= 'r')
    ax.scatter3D(B[:,0],B[:,1],B[:,2], c= 'b')
    plt.pause(0.1)


    s = init_p.shape[0]
    h = np.zeros([s,s])
    g = np.zeros(s)
    score = 0
    for i in range(cur_A.shape[0]):
        a = cur_A[i]
        b = B[i]
        point_g, point_h = gradient_hessian(init_p, a, b)
        h = h + point_h
        g = g + point_g
        score = score - fun(init_p, a, b)
    print(score)
    dp = np.linalg.pinv(-h).dot(g)
    if(np.linalg.norm(dp) < 0.0001):
        break
    cur_A = transform3d(dp, cur_A.T).T

#print(cur_p)

