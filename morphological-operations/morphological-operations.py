import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def morphological_op(image, kernel, operation):

    image = np.array(image, dtype=np.uint8)
    kernel = np.array(kernel, dtype=np.uint8)
    kh, kw = kernel.shape
    
    # Pad image to handle borders
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Create sliding windows
    windows = sliding_window_view(padded, (kh, kw))
    # windows shape: (H, W, kh, kw)
    
    # Apply kernel mask
    masked_windows = windows * kernel  # element-wise multiply
    
    if operation == "erode":
        # Erosion: all ones in neighborhood where kernel==1
        result = np.all(masked_windows == kernel, axis=(2,3)).astype(np.uint8)
    elif operation == "dilate":
        # Dilation: any one in neighborhood where kernel==1
        result = np.any(masked_windows == 1, axis=(2,3)).astype(np.uint8)
    else:
        raise ValueError("operation must be 'erode' or 'dilate'")
    
    return result.tolist()
