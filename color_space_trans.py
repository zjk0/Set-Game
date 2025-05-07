from PIL import Image
import numpy as np

def rgb_to_hsv (image_array):
    image = Image.fromarray(image_array, mode = 'RGB')
    image = image.convert("HSV")
    hsv_image_array = np.array(image)
    return hsv_image_array

def hsv_to_rgb (image_array):
    image = Image.fromarray(image_array, mode = 'HSV')
    image = image.convert("RGB")
    rgb_image_array = np.array(image)
    return rgb_image_array

def rgb_to_gray (image_array):
    image = Image.fromarray(image_array, mode = 'RGB')
    image = image.convert("L")
    gray_image_array = np.array(image)
    return gray_image_array