import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image
from PIL import ImageTk
import numpy as np
import math
import os
import cv2 as cv
import ImgProc

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

'''
@brief 获取纸牌位置和大小
'''
def get_cards_info ():
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize

    # 图像二值化
    img_bin = ImgProc.image_binarization(img_gray_resize, threshold = 180)

    contours, _ = cv.findContours(img_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 检测出连通域的边缘
    cards_info_list = []  # 纸牌中心位置列表
    for contour in contours:
        rect = cv.minAreaRect(contour)

        # 根据宽和高进行二次筛选
        if rect[1][0] > 100 and rect[1][1] > 100:
            # 整理成(row, column, w, h)的形式，其中 w < h
            if rect[1][0] < rect[1][1]:  # (w, h)的形式，较短的数值被认为是w
                card_info = (rect[0][1], rect[0][0], rect[1][0], rect[1][1])
            else:
                card_info = (rect[0][1], rect[0][0], rect[1][1], rect[1][0])
            cards_info_list.append(card_info)
            
    # 得到纸牌位置数组，并进行形状重塑和元素位置调整，调整成能与图像中纸牌一一对应的形式
    cards_info = np.array(cards_info_list)  # 转换为numpy数组
    cards_info = cards_info.astype(int)
    cards_info = cards_info.reshape(3, 4, 4)
    cards_info = cards_info[::-1, :, :]
    sort_index = np.argsort(cards_info[:, :, 1])  # 对每个坐标的列坐标进行排序，得到排序后的元素在排序前数组中的索引
    row_index = np.arange(cards_info.shape[0])[:, None]
    cards_info = cards_info[row_index, sort_index, :]

    return cards_info

'''
@brief 获取纸牌的颜色
'''
def get_color ():
    pass

'''
@brief 获取纸牌中的图形
'''
def get_appearance ():
    pass

'''
@brief 获取纸牌中的图形的数量
'''
def get_number (cards_info):
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize

    # 图像二值化
    img_bin = ImgProc.image_binarization(img_gray_resize, threshold = 180)

    # 开运算，消除条纹，使条纹变为实心，此时有实心和空心两种纹路
    img_bin = ImgProc.morphology_process(img_bin, method = "opening", dilation_se_size = 7, erosion_se_size = 9)
    show_trans_image(img_bin)

    number_matrix = np.zeros((3, 4))
    cards_info_ = cards_info.reshape(-1, 4)
    for card_info in cards_info_:
        count = 0  # 记录数值变化的次数
        first_to_second = 0  # 记录第一次变化到第二次变化的距离
        center = (card_info[0], card_info[1])

        # 减5的目的：图形与纸牌边界有一定距离，减5可以使得在遍历时，不会遍历到纸牌外的区域
        for i in range(center[0], (center[0] + int(card_info[3] / 2) - 5)):
            if i != center[0]:
                if img_bin[i, center[1]] != img_bin[i - 1, center[1]]:  # 数值变化，次数加1
                    count += 1
                    
                if count == 1:
                    first_to_second += 1

        index = np.where(np.all(cards_info == card_info, axis = -1))
        if count == 3 or count == 6:  # 次数为3（空心纹路）或6（实心纹路）
            number_matrix[index] = 3
        elif count == 4:  # 次数为4（实心纹路）
            number_matrix[index] = 2
        elif count == 1:  # 次数为1（空心纹路）
            number_matrix[index] = 1
        elif count == 2:  # 次数为2
            if first_to_second > 10:  # 距离较长（实心纹路）
                number_matrix[index] = 2
            else:  # 距离较短（空心纹路）
                number_matrix[index] = 1

    return number_matrix
    
'''
@brief 获取图形中的纹路
'''
def get_texture (cards_info):
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize
    

def test_func ():
    global img_rgb, img_gray, img_rgb_resize, img_gray_resize

    cards_info = get_cards_info()
    get_number(cards_info)
    # img_bin = ImgProc.image_binarization(img_gray_resize, threshold = 180)
    # # show_trans_image(img_bin)
    # img_bin = ImgProc.morphology_process(img_bin, method = "opening", dilation_se_size = 7, erosion_se_size = 9)
    # show_trans_image(img_bin)

'''
@brief 文件操作函数
'''
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
