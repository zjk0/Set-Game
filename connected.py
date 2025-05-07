import numpy as np
from morphology import morphology_process

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