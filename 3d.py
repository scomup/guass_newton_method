import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from math_tools import *

x = np.array([0.1,-0.1,0.1, 2.1, 2.2,-1.3])

def v2m(v):
    return np.array([[np.cos(v[2]),-np.sin(v[2]), v[0]],
            [np.sin(v[2]),np.cos(v[2]), v[1]], 
            [0,0,1]])

def m2v(m):
    return np.array([m[0,2],m[1,2],np.arctan2(m[1,0],m[0,0])])

def calcdTdx():
    A1 = np.array([0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0. ])
    A2 = np.array([0, 0, 0, 0,  0, 0, 0, 1,  0, 0, 0, 0. ])
    A3 = np.array([0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 1. ])
    A4 = np.array([0, 0, 0, 0,  0, 0,-1, 0,  0, 1, 0, 0. ])
    A5 = np.array([0, 0, 1, 0,  0, 0, 0, 0, -1, 0, 0, 0. ])
    A6 = np.array([0,-1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 0. ])
    dTdx = np.zeros([12,6])
    dTdx[:,0] = A1
    dTdx[:,1] = A2
    dTdx[:,2] = A3
    dTdx[:,3] = A4
    dTdx[:,4] = A5
    dTdx[:,5] = A6
    return dTdx

def calcdfdT(a):
    dfdT = np.empty((0,12), float)


    for i in range(a.shape[1]):
        x = a[0,i]
        y = a[1,i]
        z = a[2,i]
        tmp = np.array([[x, y, z, 1,  0, 0, 0, 0,  0, 0, 0, 0,],
                        [0, 0, 0, 0,  x, y, z, 1,  0, 0, 0, 0,],
                        [0, 0, 0, 0,  0, 0, 0, 0,  x, y, z, 1,]])
        dfdT = np.append(dfdT, tmp, axis=0)
    return dfdT

def calcRes(a,b):
    res = []
    m = 0.
    for i in range(a.shape[1]):
        x = a[0,i] - b[0,i]
        y = a[1,i] - b[1,i]
        z = a[2,i] - b[2,i]
        res.append( x)
        res.append( y)
        res.append( z)
        m += x*x + y*y + z*z
    return np.array(res), m

# f = T(x)(a) - b: objective function for the problem
# T: transform function
# x: Optimization Parameters for f

if __name__ == '__main__':

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    elements = 100
    a = (np.random.rand(elements,3)-0.5)*2
    a = a.transpose()
    b = transform3d(x, a)
    b += np.random.normal(0, 0.03, (3, elements))

    #a = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]])
    #a = a.transpose()
    #b = transform(x, a)

    dTdx = calcdTdx()
    cost =1000000000.
    last_cost = cost+1
    x_cur = np.array([0,0,0,0,0,0])
    cur_a = a
    max_loop = 20
    loop = 0
    

    while((last_cost - cost > 0.00001) or (loop > max_loop)):
        last_cost = cost
        res, cost = calcRes(cur_a, b)
        dfdT = calcdfdT(cur_a)
        J = np.dot(dfdT, dTdx)
        hessian = np.dot(J.transpose() , J)
        hessian_inv = np.linalg.inv(hessian)
        temp = -np.dot(J.transpose(), res)
        dx = np.dot(hessian_inv, temp)
        cur_a = transform3d(dx, cur_a)
        x_cur = x_cur + dx
        loop += 1

        plt.cla()
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.scatter3D(cur_a[0,:],cur_a[1,:],cur_a[2,:], c= 'r')
        ax.scatter3D(b[0,:],b[1,:],b[2,:], c= 'b')
        print(cost)
        plt.pause(0.1)
    plt.show()
