#!/usr/bin/env python
import cv2 as cv
import numpy as np
import pdb
from collections import defaultdict

def main():

    gauss_binomial_approx = {
        3: '1 2 1',
        5: '1 4 6 4 1',
        7: '1 6 15 20 15 6 1',
        9: '1 8 28 56 70 56 28 8 1',
        11: '1 10 45 120 210 252 210 120 45 10 1',
        13: '1 12 66 220 495 792 924 792 495 220 66 12 1',
        27: "1 26 325 2600 14950 65780 230230 657800 1562275 3124550 5311735 7726160 9657700 10400600 9657700 7726160 5311735 3124550 1562275 657800 230230 65780 14950 2600 325 26 1",
   }
    width_mask = {
        'gauss': {
            1: [1.0],
            },
        'flat': {
            1: np.ones(1).reshape(1,1),
            3: np.ones(3).reshape(3,1),
            5: np.ones(5).reshape(5,1),
            7: np.ones(7).reshape(7,1),
            9: np.ones(9).reshape(9,1),
            11: np.ones(11).reshape(11,1),
            13: np.ones(13).reshape(13,1),
            27: np.ones(27).reshape(27,1),
            },
    }        

    for k, v in gauss_binomial_approx.items():
        ar = np.array(v.split(), dtype=np.float64)
        ar /= sum(ar)
        width_mask['gauss'][k] = ar.reshape(len(ar), 1)

    il = cv.imread('tsukuba1.tif')
    ir = cv.imread('tsukuba2.tif')

    twoD_filters = {}
    twoD_filters['gauss'] = dict([(x, np.matmul(y, y.transpose())) for x,y in width_mask['gauss'].items() if x != 1])
    twoD_filters['flat']  = dict([(x, np.matmul(y, y.transpose())/(x**2)) for x,y in width_mask['flat'].items() if x != 1])
    for mask_type in ['gauss', 'flat']:
        for ssd_width in sorted(width_mask[mask_type].keys()):
            if ssd_width == 1:
                continue
            mask = twoD_filters[mask_type][ssd_width]
            pad = int(ssd_width/2)

            image_l = cv.copyMakeBorder(il, pad, pad, pad, pad, cv.BORDER_REPLICATE)
            image_r = cv.copyMakeBorder(ir, pad, pad, pad, pad, cv.BORDER_REPLICATE)

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
                for x_idx in range(-pad, pad+1):
                    lower_r_idx = pad + x_idx
                    lower_l_idx = lower_r_idx + col_idx
                    upper_l_idx = pad + il.shape[1] + x_idx
                    upper_r_idx = upper_l_idx - col_idx

                    for y_idx in range(ssd_width):
                        delta = image_l[y_idx:y_idx + il.shape[0],lower_l_idx:upper_l_idx,:] - image_r[y_idx:y_idx + il.shape[0],lower_r_idx:upper_r_idx,:]
                        ssd[:,col_idx:,:] += (delta * delta * mask[x_idx + pad,y_idx])

                all_ssd[col_idx,:,col_idx:,:] = ssd[:,col_idx:,:]

            # Get minimum index shifts across all shift possibilities.
            mins = np.argmin(all_ssd, axis=0)

            # Create image using median and average to combine RGB pixels.
            stat_funcs = {
            'average': np.average,
            'median': np.median,
            }

            for stat_func in stat_funcs:
                combined_mins = stat_funcs[stat_func](mins,axis=2)

                min_dist = np.min(combined_mins)

                # Normalize to 0-1
                combined_mins -= min_dist
                combined_mins /= np.max(combined_mins)

                splits = np.copy(combined_mins).reshape(combined_mins.shape[0] * combined_mins.shape[1])
                splits.sort()

                colors = [np.array(x[::-1]) for x in [
                    (0, 211, 54),
                    (163, 247, 58),
                    (254, 238, 111),
                    (236, 205, 61),
                    (247, 125, 43),
                    (210, 97, 37),
                    (146, 46, 21),
                    ]]
                
                pct = int(len(splits)/(len(colors)-1))
                bucketed = np.zeros((combined_mins.shape[0], combined_mins.shape[1], 3))
                for idx,color in enumerate(colors):
                    if idx == 0:
                        bucketed[np.where(combined_mins <= splits[pct])] = color
                    elif idx == len(colors) - 2:
                        bucketed[np.where(np.logical_and(combined_mins> splits[pct*(idx-1)], combined_mins <= splits[idx*pct] ))] = color
                        bucketed[np.where(combined_mins > splits[idx*pct])] = color
                        break
                    else:
                        bucketed[np.where(np.logical_and(combined_mins> splits[pct*(idx-1)], combined_mins <= splits[idx*pct] ))] = color
                        
                fname = 'ssd_{0}_{1}_{2}.png'.format(stat_func, mask_type, ssd_width)
                print(fname)
                # Scale to [0-255] and write to file.
                cv.imwrite(fname, bucketed.astype(np.uint8))
    return

if __name__=='__main__':
    main()
