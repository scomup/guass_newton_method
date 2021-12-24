import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from math_tools import *

# We try to optimize an exponential objective function([Magnusson 2009] 6.9), using newton's method.
# x = (Ta - b) # The transformation error (vector3)
# T(p) # The parameters(6Dof) of transformation matrix(T), We combine the 3D translation vector and the lie algebra so(3) as transformation parameters.
# c = x.T * cov^-1 * x  #ndt cost (scalar)
# f = -d1* exp(-d2/ 2* c) #Objective function (scalar)
# gradient: [Magnusson 2009] 6.12
# hessian: [Magnusson 2009] 6.13
# A ppt to help learn Newton's method
# https://www.slideshare.net/sleepy_yoshi/cvim11-3?ref=https://daily-tech.hatenablog.com/entry/2019/06/17/041356


d1 = 10.
d2 = 0.01
cov = np.eye(3)
cov_inv = np.linalg.inv(cov)

def trans_error(p, a, b):
    t = p[0:3]
    R = exp(p[3:6])
    return np.dot(R,a) + t - b

def ndt_cost(trans_error):
    return trans_error.T.dot(cov_inv.dot(trans_error))

def fun(p, a, b):
    x = trans_error(p, a, b)
    c = ndt_cost(x)
    f = -d1* np.exp(-d2 / 2 * c)
    return f

def calcdxdp(a):
    dxdp = np.array([
                [1, 0, 0, 0, a[2],-a[1]],
                [0, 1, 0,-a[2], 0, a[0]],
                [0, 0, 1, a[1],-a[0], 0]])
    return dxdp

def gradient_hessian(p, a, b):
    f = fun(p, a, b)
    x = trans_error(p, a, b)
    dxdp = calcdxdp(a)
    g = f *(-d2 * x.T.dot(cov_inv.dot(dxdp)))
    s = p.shape[0]
    h = np.zeros([s, s])
    x_cinv_dxdp = x.dot(cov_inv.dot(dxdp))
    for i in range(s):
        for j in range(s):
            h[i,j] = f * (-d2 )*  (-d2*x_cinv_dxdp[i]*x_cinv_dxdp[j] + dxdp[:,j].dot(cov_inv.dot(dxdp[:,i])))
    return g, h

p = np.array([0,0,0,0,0,0])
cur_p = p

"""
Generating test data
"""
elements = 100
A = a = (np.random.rand(elements,3)-0.5)*2
B = transform3d(np.array([0.2,0.1,-0.2, 0.5,-1.5,0.1]), A.T).T

fig = plt.figure()
ax = Axes3D(fig)
ax = Axes3D(fig)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

cur_A = A
while(True):
    plt.cla()
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_zlim(-2,2)
    ax.scatter3D(cur_A[:,0],cur_A[:,1],cur_A[:,2], c= 'r')
    ax.scatter3D(B[:,0],B[:,1],B[:,2], c= 'b')
    plt.pause(0.1)


    s = p.shape[0]
    h = np.zeros([s,s])
    g = np.zeros(s)
    score = 0
    for i in range(cur_A.shape[0]):
        a = cur_A[i]
        b = B[i]
        point_g, point_h = gradient_hessian(p, a, b)
        h = h + point_h
        g = g + point_g
        score = score - fun(p, a, b)
    print(score)
    dp = np.linalg.pinv(-h).dot(g)
    if(np.linalg.norm(dp) < 0.0001):
        break

    cur_p = cur_p + dp
    cur_A = transform3d(dp, cur_A.T).T

print(cur_p)

