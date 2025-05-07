import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image
from PIL import ImageTk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import struct
import math
import os
import cv2 as cv

from morphology import morphology_process
from connected import connected_analysis
from color_space_trans import rgb_to_hsv, hsv_to_rgb, rgb_to_gray

'''
@brief 获取图像数组
@param image: PIL的Image类对象
@return height: 图像的高（行数）
@return width: 图像的宽（列数）
@return image_array: 图像对应的数组
'''
def get_image_data (image):
    (width, height) = image.size  # 获取图像的宽和高
    image_array = np.array(image)  # 获取图像的像素数据并转化为数组
    return height, width, image_array

'''
@brief 显示转换后的图像
@param image_array: 图像数组
'''
def show_trans_image (image_array):
    global trans_image_tk

    # 转换
    trans_image = Image.fromarray(np.uint8(image_array))
    trans_image_tk = ImageTk.PhotoImage(trans_image)

    # 显示
    trans_image_label.config(image = trans_image_tk)

def image_binarization (image_array):
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize

    thres = 180
    image_array[image_array > thres] = 255
    image_array[image_array <= thres] = 0
    # show_trans_image(image_array)
    return image_array

def search_card ():
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize

    # 图像二值化
    img_bin = image_binarization(img_gray_resize)
    # show_trans_image(img_bin)

    # img_rgb_resize_copy = img_rgb_resize.copy()
    contours, _ = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 检测出连通域的边缘
    cards_pos_list = []
    for contour in contours:
        card_info = cv.minAreaRect(contour)

        # 根据长和宽进行二次筛选
        if card_info[1][0] > 100 and card_info[1][1] > 100:
            box_points = cv.boxPoints(card_info)
            box_points = np.int32(box_points)
            # cv.polylines(img_rgb_resize_copy, [box_points], isClosed = True, color = (0, 255, 0), thickness = 2)

            card_pos = (card_info[0][1], card_info[0][0])  # (row, column)的形式
            cards_pos_list.append(card_pos)

    # 得到纸牌位置数组，并进行形状重塑和元素位置调整，调整成能与图像中纸牌一一对应的形式
    cards_pos = np.array(cards_pos_list)
    cards_pos = cards_pos.astype(int)
    cards_pos = cards_pos.reshape(3, 4, 2)
    cards_pos = cards_pos[::-1, :, :]
    sort_index = np.argsort(cards_pos[:, :, 1])  # 对每个坐标的列坐标进行排序，得到排序后的元素在排序前数组中的索引
    row_index = np.arange(cards_pos.shape[0])[:, None]
    cards_pos = cards_pos[row_index, sort_index, :]

    # print(cards_pos)
    # print(cards_info.shape)

    # show_trans_image(img_rgb_resize_copy)

    return cards_pos

def test_func ():
    search_card()

def file_operation ():
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize, rgb_tk, gray_tk

    # 获取文件路径
    file_path = fd.askopenfilename()

    # 判断文件格式并得到Image对象和图像数组
    image = Image.open(file_path)
    image_gray = image.convert('L')

    # 得到原始图像数据
    _, _, img_rgb = get_image_data(image)
    _, _, img_gray = get_image_data(image_gray)

    width, height = image_gray.size
    ratio = min(800 / width, 600 / height)
    new_size = (int(width * ratio), int(height * ratio))
    image_resize = image.resize(new_size, Image.Resampling.LANCZOS)
    image_resize_gray = image_gray.resize(new_size, Image.Resampling.LANCZOS)
    rgb_tk = ImageTk.PhotoImage(image_resize)
    gray_tk = ImageTk.PhotoImage(image_resize_gray)
    _, _, img_rgb_resize = get_image_data(image_resize)
    _, _, img_gray_resize = get_image_data(image_resize_gray)

    # 显示图像
    rgb_image_label.config(image = rgb_tk)
    gray_image_label.config(image = gray_tk)
    trans_image_label.config(image = "")

if __name__ == '__main__':
    # 创建基本界面
    root = tk.Tk()
    root.title("Set Game")  # 设置界面标题
    root.grid()
    root.grid_columnconfigure(0, weight = 1)  # root的第一列会适应界面大小的改变
    root.grid_rowconfigure(0, weight = 1)  # root的第一行会适应界面大小的改变

    # 创建一个画布，画布可以被滚动条控制
    canvas = tk.Canvas(root)
    canvas.grid(row = 0, column = 0, sticky = "nsew")  # canvas会填充整个界面

    # 创建滚动条
    scrollbar_1 = ttk.Scrollbar(root, orient = "vertical", command = canvas.yview)
    scrollbar_1.grid(row = 0, column = 1, sticky = "ns")  # 纵轴方向填充
    scrollbar_2 = ttk.Scrollbar(root, orient = "horizontal", command = canvas.xview)
    scrollbar_2.grid(row = 1, column = 0, sticky = "ew")  # 横轴方向填充

    # canvas与滚动条关联
    canvas.config(yscrollcommand = scrollbar_1.set)
    canvas.config(xscrollcommand = scrollbar_2.set)

    # 创建Frame容器，用于存放各种子容器
    frame = ttk.Frame(canvas, padding = 10)
    frame.grid(row = 0, column = 0)
    canvas.create_window((0, 0), window = frame, anchor = "nw")  # 将frame嵌入canvas
    frame.bind("<Configure>", lambda event: canvas.configure(scrollregion = canvas.bbox("all")))  # 使滚动条适应frame

    # 创建存放按键的Frame容器
    button_frame = ttk.Frame(frame, padding = 10)
    button_frame.grid(row = 0, column = 0)

    # 创建“打开文件”按键
    open_file_button = ttk.Button(button_frame, text = "打开文件", command = file_operation)
    open_file_button.grid(row = 0, column = 0)

    # 创建“查找Set”按键
    search_set_button = ttk.Button(button_frame, text = "查找Set", command = test_func)
    search_set_button.grid(row = 0, column = 1)

    # 创建“退出”按键
    quit_button = ttk.Button(button_frame, text = "退出", command = root.destroy)
    quit_button.grid(row = 0, column = 2)

    # 创建图像标签
    rgb_image_label = ttk.Label(frame)
    rgb_image_label.grid(row = 1, column = 0)
    gray_image_label = ttk.Label(frame)
    gray_image_label.grid(row = 2, column = 0)
    trans_image_label = ttk.Label(frame)
    trans_image_label.grid(row = 1, column = 1, rowspan = 2)

    root.mainloop()
