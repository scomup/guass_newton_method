import numpy as np
import matplotlib.pyplot as plt
from math_tools import *

#https://www.slideshare.net/sleepy_yoshi/cvim11-3?ref=https://daily-tech.hatenablog.com/entry/2019/06/17/041356

def calcdTdx():
    A1 = np.array([0, 0, 1, 0, 0, 0])
    A2 = np.array([0, 0, 0, 0, 0, 1])
    A3 = np.array([0, -1, 0, 1, 0, 0])
    dTdx = np.zeros([6,3])
    dTdx[:,0] = A1
    dTdx[:,1] = A2
    dTdx[:,2] = A3
    return dTdx

def calcdfdT(a):
    """
    This function calculates the partial differential matrix of f near the T(0) 
    f = T(a) - b
    T = | t1, t2, t3 |  a = [u, v, 1]
        | t4, t5, t6 |
        | t7, t8, t9 | 
    try find: df/dT|T(0)
    """
    dfdT = np.empty((0,6), float)
    for i in range(a.shape[1]):
        x = a[0,i]
        y = a[1,i]
        tmp = np.array([[x, y, 1,  0, 0, 0],
                        [0, 0, 0,  x, y, 1]])
        dfdT = np.append(dfdT, tmp, axis=0)
    return dfdT

def calcRes(a,b):
    res = []
    m = 0.
    for i in range(a.shape[1]):
        x = a[0,i] - b[0,i]
        y = a[1,i] - b[1,i]
        res.append(x)
        res.append(y)
        m += x*x + y*y
    return np.array(res), m

# f = T(x)(a) - b: objective function for the problem
# T: transform function
# x: Optimization Parameters for f
if __name__ == '__main__':
    x = np.array([-0.3,0.2,np.pi])

    elements = 100
    a = (np.random.rand(elements,2)-0.5)*2
    a = a.transpose()
    b = transform2d(x, a)
    b += np.random.normal(0, 0.03, (2, elements))

    dTdx = calcdTdx()
    cost =1000000000.
    last_cost = cost+1
    x_cur = np.array([0,0,0.])
    cur_a = a
    max_loop = 20
    loop = 0
    while((last_cost - cost > 0.0001) and (loop < max_loop)):
        last_cost = cost
        res, cost = calcRes(cur_a, b)
        dfdT = calcdfdT(cur_a)
        J = np.dot(dfdT, dTdx)
        hessian = np.dot(J.transpose() , J)
        hessian_inv = np.linalg.inv(hessian)
        temp = -np.dot(J.transpose(), res)
        dx = np.dot(hessian_inv, temp)
        cur_a = transform2d(dx, cur_a)
        x_cur = x_cur + dx
        loop += 1

        plt.cla()
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.scatter(cur_a[0,:], cur_a[1,:], c= 'r')
        plt.scatter(b[0,:], b[1,:], c= 'b')
        print(cost)
        plt.pause(0.1)
    plt.show()
