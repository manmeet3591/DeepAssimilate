# Function to generate random noise images
def generate_random_image(seed, img_size=(256, 256)):
    np.random.seed(seed)
    random_image = np.random.rand(*img_size)
    return random_image

# Bicubic interpolation for downscaling
def bicubic_downscale(image, scale_factor):
    height, width = image.shape
    new_size = (int(width // scale_factor), int(height // scale_factor))
    downscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return downscaled_image

# Bicubic upscaling to original size
def bicubic_upscale(image, original_size):
    upscaled_image = cv2.resize(image, original_size, interpolation=cv2.INTER_CUBIC)
    return upscaled_image

# Function to create "stations" image with 90% data missing
def create_stations_image(original_image, gap_ratio=0.9):
    mask = np.random.rand(*original_image.shape) < gap_ratio
    stations_image = original_image.copy()
    stations_image[mask] = np.nan  # Mask out 90% of the data
    return stations_image, mask  # Also return the mask for loss calculation

def nearest_neighbor_resize(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    return resized_image
