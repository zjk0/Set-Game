import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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