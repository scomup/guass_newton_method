import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from math_tools import *


def calcdfdx(a):
    dfdx = np.empty((0,3), float)
    for i in range(a.shape[1]):
        x = a[0,i]
        y = a[1,i]
        z = a[2,i]
        tmp = np.array([[x,y,z]])
        dfdx = np.append(dfdx, tmp, axis=0)
    return dfdx

def calcRes(a,plane):
    res = np.empty((0), float)
    m = 0.
    for i in range(a.shape[1]):
        d = np.array([plane[0:3].dot(a[:,i]) + 1])
        res = np.append(res, d, axis=0)
        m += d*2

    return res, m



if __name__ == '__main__':
    
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    #x = np.array([0, 0.2, 0.4, -0.5, 0.2, 0.4])
    x = np.array([0.0, 0.0, -1, -0.3, 0.2, -0.1])

    mean = np.array([0, 0, 0])

    cov = np.array([[1, 0, 0], 
                    [0, 0.5, 0],
                    [0, 0, 0.01]])


    elements = 100
    a = np.random.default_rng().multivariate_normal(mean, cov, elements).T
    a = transform3d(x, a)
    x_cur = np.array([0,0,0])

    max_loop = 200
    loop = 0
    cost =1000000000.
    last_cost = cost+1

    while((last_cost - cost > 0.0001) and (loop < max_loop)):
        last_cost = cost
        res, cost = calcRes(a, x_cur)
        J = calcdfdx(a)
        hessian = np.dot(J.transpose() , J)
        hessian_inv = np.linalg.inv(hessian)
        temp = -np.dot(J.transpose(), res)
        dx = np.dot(hessian_inv, temp)
        x_cur = x_cur + dx
        loop += 1
        plt.cla()
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.scatter3D(a[0,:],a[1,:],a[2,:], c= 'r')
        xx, yy = np.meshgrid(np.arange(-2,2), np.arange(-2,2))
        z = (-x_cur[0] * xx - x_cur[1] * yy - 1)/x_cur[2]

        # plot the plane
        ax.plot_surface(xx, yy, z, alpha=0.5)

        print(cost)
        plt.pause(0.1)
    print(x_cur)
    plt.show()

