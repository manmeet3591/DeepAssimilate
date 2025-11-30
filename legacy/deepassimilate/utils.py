import numpy as np
import cv2

def generate_random_image(seed, img_size=(256, 256)):
    np.random.seed(seed)
    return np.random.rand(*img_size)

def bicubic_downscale(image, scale_factor):
    height, width = image.shape
    new_size = (int(width // scale_factor), int(height // scale_factor))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

def bicubic_upscale(image, original_size):
    return cv2.resize(image, original_size, interpolation=cv2.INTER_CUBIC)


import numpy as np
import cv2

def nearest_neighbor_resize_with_nan(image, target_size):
    """
    Resize an image using nearest neighbor interpolation while preserving NaN values.
    Args:
        image (np.ndarray): Input image with potential NaN values (C, H, W).
        target_size (tuple): Target size (H, W).
    Returns:
        np.ndarray: Resized image with NaN values preserved.
    """
    # Create a mask for NaN values
    nan_mask = np.isnan(image)

    # Replace NaNs with a placeholder value (e.g., 0) before resizing
    image_filled = np.where(nan_mask, 0, image)

    # Perform nearest neighbor resizing
    channels = [cv2.resize(image_filled[c], target_size, interpolation=cv2.INTER_NEAREST)
                for c in range(image_filled.shape[0])]

    # Resize the NaN mask separately
    resized_nan_mask = [cv2.resize(nan_mask[c].astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
                        for c in range(nan_mask.shape[0])]

    # Stack channels and mask
    resized_image = np.stack(channels, axis=0)
    resized_nan_mask = np.stack(resized_nan_mask, axis=0).astype(bool)

    # Restore NaN values in the resized image
    resized_image_with_nan = np.where(resized_nan_mask, np.nan, resized_image)

    return resized_image_with_nan
