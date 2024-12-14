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
