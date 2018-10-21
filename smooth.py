#!/usr/bin/env python

import cv2
import numpy as np
import pdb
import sys
import time

def main(image_file_name):
    original = cv2.imread(image_file_name)

    oneD_filters = {}
    gauss_binomial_approx = {
        3: '1 2 1',
        7: '1 6 15 20 15 6 1',
        11: '1 10 45 120 210 252 210 120 45 10 1',
        13: '1 12 66 220 495 792 924 792 495 220 66 12 1'
        }

    for k, v in gauss_binomial_approx.items():
        ar = np.array(v.split(), dtype=np.float64)
        ar /= sum(ar)
        oneD_filters[k] = ar.reshape(len(ar), 1)

    twoD_filters = dict([(x, np.matmul(y, y.transpose())) for x,y in oneD_filters.items()])

    # for sigma in [1,2,3]:
    #     fname = 'guass_{}.png'.format(sigma)
    #     smoothed = cv2.GaussianBlur(original, (0,0), sigma)
    #     cv2.imwrite(fname, smoothed)
    #     print(fname)

    for width, filt in twoD_filters.items():
        t0 = time.time()
        pad = int(width/2)
        image = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        smoothed = np.zeros(original.shape)
        img_height = image.shape[0]
        img_width = image.shape[1]
        for x_idx in range(width):
            for y_idx in range(width):
                smoothed += image[x_idx:img_height - width + x_idx + 1,
                                  y_idx:img_width -  width + y_idx + 1,
                                  :] * filt[x_idx,y_idx]
        fname = 'smoothed_{0}.png'.format(width)
        cv2.imwrite(fname, smoothed)
        print("{0} {1}".format(fname, time.time() - t0))

    for width in [3,7,11,13]:
        t0 = time.time()
        pad = int(width/2)
        image = cv2.copyMakeBorder(original, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        smoothed = np.zeros(original.shape)
        img_height = image.shape[0]
        img_width = image.shape[1]
        filt = oneD_filters[width]
        for idx in range(width):
            smoothed += image[idx:img_height - width + idx + 1,
                              pad:img_width - pad,
                              :] * filt[idx]

        image = cv2.copyMakeBorder(smoothed, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        smoothed = np.zeros(original.shape)
        for idx in range(width):
            smoothed += image[pad:img_height - pad,
                              idx:img_width - width + idx + 1,
                              :] * filt[idx]
        fname = 'smoothed_{0}_1D.png'.format(width)
        cv2.imwrite(fname, smoothed)
        print("{0} {1}".format(fname, time.time() - t0))

    return

if __name__=='__main__':
    main(sys.argv[1])
    
