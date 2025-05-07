import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

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
@brief 图像二值化
'''
def image_binarization (image_array, threshold):
    image_array[image_array > threshold] = 255
    image_array[image_array <= threshold] = 0
    return image_array

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

'''
@brief 连通域检测
'''
def connected_analysis (image_array):
    image_array = morphology_process(image_array, "opening", dilation_se_size = 5, erosion_se_size = 5)

    label_value = 0
    values = np.zeros(4, dtype = np.int32)
    labels = np.zeros(4, dtype = np.int32)
    equ_label = np.arange(image_array.shape[0] * image_array.shape[1])

    expand_array = np.pad(image_array, pad_width = 1, mode = 'constant', constant_values = 0)

    rows = expand_array.shape[0]
    columns = expand_array.shape[1]
    array_with_label = np.zeros((rows, columns, 2), dtype = np.int32)
    array_with_label[:, :, 0] = expand_array
    array_with_label[:, :, 1] = -1  # 表示未打标签

    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            if array_with_label[i, j, 0] == 255:
                values[0] = array_with_label[i, j - 1, 0]  # 左
                values[1] = array_with_label[i - 1, j - 1, 0]  # 左上
                values[2] = array_with_label[i - 1, j, 0]  # 上
                values[3] = array_with_label[i - 1, j + 1, 0]  # 右上
                labels[0] = array_with_label[i, j - 1, 1]
                labels[1] = array_with_label[i - 1, j - 1, 1]
                labels[2] = array_with_label[i - 1, j, 1]
                labels[3] = array_with_label[i - 1, j + 1, 1]

                if np.count_nonzero(values == 255) == 0:  # 周围没有前景点
                    array_with_label[i, j, 1] = label_value
                    label_value += 1
                else:  # 周围有前景点
                    if np.count_nonzero(labels != -1) == 0:  # 前景点都没有标签
                        array_with_label[i, j, 1] = label_value
                        array_with_label[i, j - 1, 1] = label_value if values[0] == 255 else (-1)
                        array_with_label[i - 1, j - 1, 1] = label_value if values[1] == 255 else (-1)
                        array_with_label[i - 1, j, 1] = label_value if values[2] == 255 else (-1)
                        array_with_label[i - 1, j + 1, 1] = label_value if values[3] == 255 else (-1)
                        label_value += 1
                    else:  # 有的前景点有标签
                        labels_1 = labels[labels != -1]
                        for k in range(labels_1.shape[0]):
                            # 寻找最小等价标签
                            while equ_label[labels_1[k]] != labels_1[k]:
                                labels_1[k] = equ_label[labels_1[k]]

                        min_label = np.min(labels_1)
                        
                        # 打标签
                        array_with_label[i, j, 1] = min_label
                        array_with_label[i, j - 1, 1] = min_label if (labels[0] == -1 and values[0] == 255) else labels[0]
                        array_with_label[i - 1, j - 1, 1] = min_label if (labels[1] == -1 and values[1] == 255) else labels[1]
                        array_with_label[i - 1, j, 1] = min_label if (labels[2] == -1 and values[2] == 255) else labels[2]
                        array_with_label[i - 1, j + 1, 1] = min_label if (labels[3] == -1 and values[3] == 255) else labels[3]

                        # 修改等价标签
                        equ_label[labels_1] = min_label
    
    # 第二次遍历
    for i in range(1, rows - 1):
        for j in range(1, columns - 1):
            if array_with_label[i, j, 0] == 255:
                # 寻找最小等价标签
                while equ_label[array_with_label[i, j, 1]] != array_with_label[i, j, 1]:
                    array_with_label[i, j, 1] = equ_label[array_with_label[i, j, 1]]

    # 彩色标注
    temp_array = np.zeros((rows, columns, 3), dtype = np.int32)
    rgb_array = np.random.randint(0, 256, (np.max(array_with_label[:, :, 1]) + 1, 3))
    for i in range(np.max(array_with_label[:, :, 1]) + 1):
        temp_array[array_with_label[:, :, 1] == i] = rgb_array[i]

    result = temp_array[1 : rows - 1, 1 : columns - 1]
    return result