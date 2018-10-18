#!/usr/bin/env python

import numpy as np
import pdb
import scipy.io as sio

def main():
    mat = sio.loadmat('calibration-dataset-1.mat')
    X = mat['X']
    x = mat['x2']

    N = X.shape[1]
    D           = np.zeros((2*N, 12), np.float64)
    D[::2,:3]   = X.transpose()
    D[::2,3]    = 1.0
    D[::2,8:11] = (X * x[0] * -1.0).transpose()
    D[::2,11]   = x[0] * -1.0

    D[1::2,4:7]  = X.transpose()
    D[1::2,7]    = 1.0
    D[1::2,8:11] = (X * x[1] * -1.0).transpose()
    D[1::2,11]   = x[1] * -1.0

    Dpi = np.linalg.pinv(D)

    f = np.zeros(N*2)
    f[::2] = x[0]
    f[1::2] = x[1]

    C = np.matmul(Dpi,f).reshape(3,4)

    Xh = np.zeros((N,4))
    Xh[:,:3] = X.transpose()
    Xh[:,3:] = 1.0
    Xh = Xh.reshape(20,4,1)
    x2 = np.matmul(C,Xh)     
    x2 = x2.reshape(N,3)/ x2[:,2].reshape(N)[:,None]

    print("C = {}".format(C))
    print("initial camera coordinates = {}".format(x.transpose()))
    print("calculated camera coordinates = {}".format(x2))
    return C

if __name__ == '__main__':
    main()
    
