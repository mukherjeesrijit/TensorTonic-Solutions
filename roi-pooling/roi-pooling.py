import math
import numpy as np

def roi_pool(feature_map, rois, output_size):
    
    """
    Apply ROI Pooling to extract fixed-size features.
    """

    feature_map = np.array(feature_map)
    rois = np.array(rois)

    all_results = []

    for roi in rois:
        output = np.zeros((output_size, output_size))
        for i,j in np.ndindex(output_size, output_size):
            x1, y1, x2, y2 = roi
            roi_h = y2-y1
            roi_w = x2-x1
            h_s = int(y1 + np.floor(i*roi_h/output_size))
            h_e = int(y1 + np.floor((i+1)*roi_h/output_size))
            if h_e == h_s: h_e = h_s + 1
            w_s = int(x1 + np.floor(j*roi_w/output_size))
            w_e = int(x1 + np.floor((j+1)*roi_w/output_size))
            if w_e == w_s: w_e = w_s + 1
            bin_slice = feature_map[h_s:h_e, w_s:w_e]
            if bin_slice.size > 0:
                output[i, j] = np.max(bin_slice)
            else:
                output[i, j] = 0 # Or another default value
        all_results.append(output.tolist())
    return all_results


    

            
            
            