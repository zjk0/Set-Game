import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
import cv2 as cv

'''
@brief RGB空间转换到HSV空间
'''
def rgb_to_hsv (image_array):
    image = Image.fromarray(image_array, mode = 'RGB')
    image = image.convert("HSV")
    hsv_image_array = np.array(image)
    return hsv_image_array

'''
@brief HSV空间转换到RGB空间
'''
def hsv_to_rgb (image_array):
    image = Image.fromarray(image_array, mode = 'HSV')
    image = image.convert("RGB")
    rgb_image_array = np.array(image)
    return rgb_image_array

'''
@brief RGB空间转换到灰度空间
'''
def rgb_to_gray (image_array):
    image = Image.fromarray(image_array, mode = 'RGB')
    image = image.convert("L")
    gray_image_array = np.array(image)
    return gray_image_array

'''
@brief 灰度空间转换到HSV空间
'''
def gray_to_hsv (image_array):
    image = Image.fromarray(image_array, mode = 'L')
    image = image.convert("HSV")
    hsv_image_array = np.array(image)
    return hsv_image_array

'''
@brief HSV空间转换到灰度空间
'''
def hsv_to_gray (image_array):
    image = Image.fromarray(image_array, mode = 'HSV')
    image = image.convert('L')
    gray_image_array = np.array(image)
    return gray_image_array

'''
@brief 图像二值化
'''
# def image_binarization (image_array, s_threshold, v_threshold):
#     hsv = cv.cvtColor(image_array, cv.COLOR_BGR2HSV)
#     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
#     img_eq = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
#     img_bin = cv.cvtColor(img_eq, cv.COLOR_BGR2GRAY)
#     _, img_bin = cv.threshold(img_bin, 160, 255, cv.THRESH_BINARY)

#     img_bin = cv.GaussianBlur(image_array, (3, 3), 0)
#     _, img_bin = cv.threshold(img_bin, 0, 255, cv.THRESH_OTSU)

#     invG = 1 / 0.5   # gamma<1 拉亮
#     table = (np.arange(256) / 255.0) ** invG * 255
#     lut = table.astype('uint8')
#     gray2 = cv.LUT(image_array, lut)
#     _, img_bin = cv.threshold(gray2, 50, 255, cv.THRESH_BINARY)

#     hsv_img = rgb_to_hsv(image_array)
#     img_bin = np.zeros_like(hsv_img)
#     img_bin[(hsv_img[:, :, 1] > s_threshold) | (hsv_img[:, :, 2] <= v_threshold), :] = [0, 0, 0]  # 黑色
#     img_bin[(hsv_img[:, :, 1] <= s_threshold) | (hsv_img[:, :, 2] > v_threshold), :] = [0, 0, 255]  # 白色
#     img_bin = hsv_to_rgb(img_bin)
#     img_bin = rgb_to_gray(img_bin)

#     image_array[image_array > threshold] = 255
#     image_array[image_array <= threshold] = 0
#     return img_bin

'''
@brief 形态学处理
'''
def morphology_process (image_array, method, dilation_se_size = 3, erosion_se_size = 3):
    # 获取图像行数和列数
    rows = image_array.shape[0]
    columns = image_array.shape[1]

    # 膨胀或者腐蚀
    if method == "dilation" or method == "erosion":
        # 获取结构元大小
        if method == "dilation":
            se_size = dilation_se_size
        elif method == "erosion":
            se_size = erosion_se_size

        # 边缘填充，采用零填充
        new_rows = rows + se_size - 1
        new_columns = columns + se_size - 1
        expand_array = np.zeros((new_rows, new_columns))
        offset = int((se_size - 1) / 2)
        expand_array[offset : rows + offset, offset : columns + offset] = np.copy(image_array)

        # 形态学处理
        se = np.ones((se_size, se_size))
        windows = sliding_window_view(expand_array, (se_size, se_size))
        if method == "dilation":
            result = np.max(np.multiply(windows[:, :], se), axis = (-2, -1))
        elif method == "erosion":
            result = np.min(np.multiply(windows[:, :], se), axis = (-2, -1))
    # 开运算或者闭运算
    elif method == "opening" or method == "closing":
        # 获取两个结构元大小
        if method == "opening":
            se_size_1 = erosion_se_size
            se_size_2 = dilation_se_size
        elif method == "closing":
            se_size_1 = dilation_se_size
            se_size_2 = erosion_se_size

        # 第一次边缘填充，采用零填充
        new_rows = rows + se_size_1 - 1
        new_columns = columns + se_size_1 - 1
        expand_array_1 = np.zeros((new_rows, new_columns))
        offset = int((se_size_1 - 1) / 2)
        expand_array_1[offset : rows + offset, offset : columns + offset] = np.copy(image_array)

        # 第一次形态学处理
        se_1 = np.ones((se_size_1, se_size_1))
        windows_1 = sliding_window_view(expand_array_1, (se_size_1, se_size_1))
        if method == "opening":  # 先进行腐蚀
            result = np.min(np.multiply(windows_1[:, :], se_1), axis = (-2, -1))
        elif method == "closing":  # 先进行膨胀
            result = np.max(np.multiply(windows_1[:, :], se_1), axis = (-2, -1))

        # 第二次边缘填充，采用零填充
        new_rows = rows + se_size_2 - 1
        new_columns = columns + se_size_2 - 1
        expand_array_2 = np.zeros((new_rows, new_columns))
        offset = int((se_size_2 - 1) / 2)
        expand_array_2[offset : rows + offset, offset : columns + offset] = np.copy(result)

        # 第二次形态学处理
        se_2 = np.ones((se_size_2, se_size_2))
        windows_2 = sliding_window_view(expand_array_2, (se_size_2, se_size_2))
        if method == "opening":  # 后进行膨胀
            result = np.max(np.multiply(windows_2[:, :], se_2), axis = (-2, -1))
        elif method == "closing":  # 后进行腐蚀
            result = np.min(np.multiply(windows_2[:, :], se_2), axis = (-2, -1))

    return result