import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from math_tools import *

# We try to optimize an exponential objective function([Magnusson 2009] 6.9), using newton's method.
# x = (T*dT*a - b) # The transformation error (vector3)
# T(p) # The parameters(6Dof) of transformation matrix(T), We combine the 3D translation vector and the lie algebra so(3) as transformation parameters.
# c = x.T * cov^-1 * x  #ndt cost (scalar)
# f = -d1* exp(-d2/ 2* c) #Objective function (scalar)
# gradient: [Magnusson 2009] 6.12
# hessian: [Magnusson 2009] 6.13
# A ppt to help learn Newton's method
# https://www.slideshare.net/sleepy_yoshi/cvim11-3?ref=https://daily-tech.hatenablog.com/entry/2019/06/17/041356


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

def ndt_cost(trans_error):
    return trans_error.T.dot(cov_inv.dot(trans_error))

def fun(init_p, a, b):
    x = trans_error(init_p, a, b)
    c = ndt_cost(x)
    f = -d1* np.exp(-d2 / 2 * c)
    return f

def calcdxdp(init_p, a):
    T = x2m(init_p)
    x = a[0]
    y = a[1]
    z = a[2] 
    dxdp = np.array([[T[0,0], T[0,1], T[0,2], -T[0,1]*z + T[0,2]*y, T[0,0]*z - T[0,2]*x, -T[0,0]*y + T[0,1]*x],
                    [T[1,0], T[1,1], T[1,2], -T[1,1]*z + T[1,2]*y, T[1,0]*z - T[1,2]*x, -T[1,0]*y + T[1,1]*x],
                    [T[2,0], T[2,1], T[2,2], -T[2,1]*z + T[2,2]*y, T[2,0]*z - T[2,2]*x, -T[2,0]*y + T[2,1]*x],])



    return dxdp

def calcdx2dp2(a):
    x1 = a[0]
    x2 = a[1]
    x3 = a[2]
    dx2dp2 = np.zeros([18,6])
    dx2dp2[9 : 12, 3] = np.array([0,-x2,-x3])
    dx2dp2[12: 15, 3] = np.array([0,x1,0])
    dx2dp2[15: 18, 3] = np.array([0,0,x1])
    dx2dp2[9 : 12, 4] = np.array([0,x1,0])
    dx2dp2[12: 15, 4] = np.array([-x1,0,-x3])
    dx2dp2[15: 18, 4] = np.array([0,0,x2])
    dx2dp2[9 : 12, 5] = np.array([0,0,x1])
    dx2dp2[12: 15, 5] = np.array([0,0,x2])
    dx2dp2[15: 18, 5] = np.array([-x1,-x2,0])
    return dx2dp2


def gradient_hessian(init_p, a, b):
    f = fun(init_p, a, b)
    x = trans_error(init_p, a, b)
    dxdp = calcdxdp(init_p, a)
    dx2dp2 = calcdx2dp2(a)
    g = f *(-d2 * x.T.dot(cov_inv.dot(dxdp)))
    s = init_p.shape[0]
    h = np.zeros([s, s])
    x_cinv_dxdp = x.dot(cov_inv.dot(dxdp))
    for i in range(s):
        for j in range(s):
            x_cinv_dx2dpij = x.dot(cov_inv.dot(dx2dp2[i*3: i*3 + 3, j]))
            h[i,j] = f * (-d2 )*  (-d2*x_cinv_dxdp[i]*x_cinv_dxdp[j]  +x_cinv_dx2dpij +  dxdp[:,j].dot(cov_inv.dot(dxdp[:,i])))
    return g, h

"""
Generating test data
"""
elements = 100
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

