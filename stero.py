#!/usr/bin/env python
import cv2 as cv
import numpy as np
import pdb

def main():

    gauss_binomial_approx = {
        3: '1 2 1',
        5: '1 4 6 4 1',
        7: '1 6 15 20 15 6 1',
        9: '1 8 28 56 70 56 28 8 1',
        11: '1 10 45 120 210 252 210 120 45 10 1',
        13: '1 12 66 220 495 792 924 792 495 220 66 12 1',
    }
    width_mask = {
        'gauss': {
            1: [1.0],
            },
        'flat': {
            1: [1.0],
            3: np.ones(3),
            5: np.ones(5),
            7: np.ones(7),
            9: np.ones(9),
            },
    }        

    for k, v in gauss_binomial_approx.items():
        ar = np.array(v.split(), dtype=np.float64)
        ar /= sum(ar)
        width_mask['gauss'][k] = ar.reshape(len(ar), 1)

    stat_funcs = {
        'average': np.average,
        'median': np.median,
        }

    il = cv.imread('tsukuba1.tif')
    ir = cv.imread('tsukuba2.tif')

    for mask_type in ['gauss', 'flat']:
        for ssd_width in sorted(width_mask[mask_type].keys()):
            mask = width_mask[mask_type][ssd_width]
            pad = int(ssd_width/2)

            image_l = cv.copyMakeBorder(il, 0, 0, pad, pad, cv.BORDER_REPLICATE)
            image_r = cv.copyMakeBorder(ir, 0, 0, pad, pad, cv.BORDER_REPLICATE)

            # For each possible shift of pixels (ie, for each column)
            # create an array for each pixel on the image.
            # Initialize all values to infinity, since we will be
            # overlaying smaller calculated SSD values, and taking
            # minimum values across all shifts.
            all_ssd = np.full([il.shape[1]] + list(il.shape), np.inf)

            for col_idx in range(il.shape[1]):

                # create array in which to accumulate SSD window values
                ssd = np.zeros(il.shape)

                # Calculate SSD convolution over entire image at each step.
                # Accumulate values.
                # Assume all shifts must be rightward from right image to left image.
                # Don't calculate for right image pixels that would be off the left image.
                for idx in range(-pad, pad+1):
                    lower_r_idx = pad + idx
                    lower_l_idx = lower_r_idx + col_idx
                    upper_l_idx = pad + il.shape[1] + idx
                    upper_r_idx = upper_l_idx - col_idx
                    delta = image_l[:,lower_l_idx:upper_l_idx,:] - image_r[:,lower_r_idx:upper_r_idx,:]
                    ssd[:,col_idx:,:] += (delta * delta * mask[idx + pad])
                all_ssd[col_idx,:,col_idx:,:] = ssd[:,col_idx:,:]

            # Get minimum index shifts across all shift possibilities.
            mins = np.argmin(all_ssd, axis=0)

            # Create image using median and average to combine RGB pixels.
            for stat_func in stat_funcs:
                combined_mins = stat_funcs[stat_func](mins,axis=2)

                min_dist = np.min(combined_mins)

                # Normalize to 0-1
                combined_mins -= min_dist
                combined_mins /= np.max(combined_mins)

                fname = 'distances_{0}_{1}_{2}.png'.format(stat_func, mask_type, ssd_width)
                print(fname)
                # Scale to [0-255] and write to file.
                cv.imwrite(fname, (256 * combined_mins).astype(np.uint8))

    return

if __name__=='__main__':
    main()
    
