#!/usr/bin/env python
import cv2 as cv
import numpy as np
import pdb

def main():

    il = cv.imread('tsukuba1.tif')
    ir = cv.imread('tsukuba2.tif')
    
    # black = np.zeros((il.shape[1]))
    # for x in range(25):
    #     il[x + 100, :, 0] = black
    #     il[x + 100, :, 1] = black
    #     il[x + 100, :, 2] = black
    # cv.imwrite('tsukuba1_mod.png', il)

    ssd_width = 5
    pad = int(ssd_width/2)
    
    image_l = cv.copyMakeBorder(il, 0, 0, pad, pad, cv.BORDER_REPLICATE)
    image_r = cv.copyMakeBorder(ir, 0, 0, pad, pad, cv.BORDER_REPLICATE)

    pdb.set_trace()
    all_ssd = np.full([il.shape[1]] + list(il.shape), np.inf)
    for col_idx in range(il.shape[1]):
        ssd = np.zeros(il.shape)
        for idx in range(-pad, pad+1):
            lower_r_idx = pad + idx
            upper_r_idx = pad + il.shape[1] + idx - col_idx
            lower_l_idx = pad + idx + col_idx
            upper_l_idx = pad + il.shape[1] + idx
            delta = image_l[:,lower_l_idx:upper_l_idx,:] - image_r[:,lower_r_idx:upper_r_idx,:]
            ssd[:,col_idx:,:] += (delta * delta)
        all_ssd[col_idx,:,col_idx:,:] = ssd[:,col_idx:,:]

    mins = np.argmin(all_ssd, axis=0)
    avg_mins = np.average(mins,axis=2)
    
    # normalize to [0-1]
    min_dist = np.min(avg_mins)
    avg_mins -= min_dist
    avg_mins /= np.max(avg_mins)
    cv.imwrite('distances.png', (255 * avg_mins).astype(np.uint8))

    return

if __name__=='__main__':
    main()
    
