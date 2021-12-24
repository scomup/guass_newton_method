import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from math_tools import *

# f (Ta - b).T C^-1 (Ta - b)  
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

def calcdfdT(s, a):
    w = s.T
    dfdT = np.empty((0,12), float)
    for i in range(a.shape[1]):
        x = a[0,i]
        y = a[1,i]
        z = a[2,i]
        tmp = np.array([[w[0,0] * x, w[0,0] * y, w[0,0] * z, w[0,0] * 1,  
                         w[0,1] * x, w[0,1] * y, w[0,1] * z, w[0,1] * 1,
                         w[0,2] * x, w[0,2] * y, w[0,2] * z, w[0,2] * 1],
                        [w[1,0] * x, w[1,0] * y, w[1,0] * z, w[1,0] * 1,  
                         w[1,1] * x, w[1,1] * y, w[1,1] * z, w[1,1] * 1,
                         w[1,2] * x, w[1,2] * y, w[1,2] * z, w[1,2] * 1],
                        [w[2,0] * x, w[2,0] * y, w[2,0] * z, w[2,0] * 1,  
                         w[2,1] * x, w[2,1] * y, w[2,1] * z, w[2,1] * 1,
                         w[2,2] * x, w[2,2] * y, w[2,2] * z, w[2,2] * 1]])
        dfdT = np.append(dfdT, tmp, axis=0)
    return dfdT

def calcRes(s,a,b):
    #res = []
    res = np.empty((0), float)
    m = 0.
    for i in range(a.shape[1]):
        d = a[:,i] - b
        h = s.T.dot(d)
        res = np.append(res, h, axis=0)
        m += h.T.dot(h)

    return res, m



if __name__ == '__main__':
    
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    #x = np.array([0, 0.2, 0.4, -0.5, 0.2, 0.4])
    x = np.array([0.2, -0.3, 0.4, 0.2, 0.8, -0.5])

    mean = np.array([0, 0, 0])

    cov = np.array([[1, 0, 0], 
                    [0, 0.5, 0],
                    [0, 0, 0.01]])


    elements = 100
    a = np.random.default_rng().multivariate_normal(mean, cov, elements).T

    cloud = np.random.default_rng().multivariate_normal(mean, cov, 1000).T
    cloud = transform3d(x, cloud)
    cov = np.cov(cloud)
    cov_inv = np.linalg.inv(cov)
    s = np.linalg.cholesky(cov_inv)
    b = np.mean(cloud, axis=1)

    p = np.array([0, 0, 0])

    r = p.T.dot(cov_inv.dot(p))

    dTdx = calcdTdx()
    cost =1000000000.
    last_cost = cost+1
    x_cur = np.array([0,0,0,0,0,0])
    cur_a = a
    max_loop = 200
    loop = 0
    
    while((last_cost - cost > 0.0001) and (loop < max_loop)):
        last_cost = cost
        res, cost = calcRes(s, cur_a, b)
        dfdT = calcdfdT(s, cur_a)
        J = np.dot(dfdT, dTdx)
        hessian = np.dot(J.transpose() , J)
        hessian_inv = np.linalg.inv(hessian)
        g = np.dot(J.transpose(), res)
        dx = np.dot(-hessian_inv, g)
        cur_a = transform3d(dx, cur_a)
        x_cur = x_cur + dx
        loop += 1

        plt.cla()
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.scatter3D(cur_a[0,:],cur_a[1,:],cur_a[2,:], c= 'r')
        ax.scatter3D(cloud[0,:],cloud[1,:],cloud[2,:], c= 'b')
        print(cost)
        plt.pause(0.1)
    print(x_cur)
    plt.show()

