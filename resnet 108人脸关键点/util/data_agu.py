# -*-coding:utf-8-*-
# date:2019-05-20
# Author: Eric.Lee
# function: face rot imageaug

import cv2
import numpy as np
import random

# 图像翻转后关键点的位置变换（98个关键点）
flip_landmarks_dict = {
    0: 32, 1: 31, 2: 30, 3: 29, 4: 28, 5: 27, 6: 26, 7: 25, 8: 24, 9: 23, 10: 22, 11: 21, 12: 20, 13: 19, 14: 18,
    15: 17,
    16: 16, 17: 15, 18: 14, 19: 13, 20: 12, 21: 11, 22: 10, 23: 9, 24: 8, 25: 7, 26: 6, 27: 5, 28: 4, 29: 3, 30: 2,
    31: 1, 32: 0,
    33: 46, 34: 45, 35: 44, 36: 43, 37: 42, 38: 50, 39: 49, 40: 48, 41: 47,
    46: 33, 45: 34, 44: 35, 43: 36, 42: 37, 50: 38, 49: 39, 48: 40, 47: 41,
    60: 72, 61: 71, 62: 70, 63: 69, 64: 68, 65: 75, 66: 74, 67: 73,
    72: 60, 71: 61, 70: 62, 69: 63, 68: 64, 75: 65, 74: 66, 73: 67,
    96: 97, 97: 96,
    51: 51, 52: 52, 53: 53, 54: 54,
    55: 59, 56: 58, 57: 57, 58: 56, 59: 55,
    76: 82, 77: 81, 78: 80, 79: 79, 80: 78, 81: 77, 82: 76,
    87: 83, 86: 84, 85: 85, 84: 86, 83: 87,
    88: 92, 89: 91, 90: 90, 91: 89, 92: 88,
    95: 93, 94: 94, 93: 95
}


# 非形变处理:将图像按长宽比resize，然后进行pad
def letterbox(img_, img_size=256, mean_rgb=(128, 128, 128)):
    shape_ = img_.shape[:2]  # shape = [height, width]
    ratio = float(img_size) / max(shape_)  # ratio  = old / new
    new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
    dw_ = (img_size - new_shape_[0]) / 2  # width padding
    dh_ = (img_size - new_shape_[1]) / 2  # height padding
    top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
    left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
    # resize img
    img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

    img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT,
                               value=mean_rgb)  # padded square

    return img_a


def img_agu_channel_same(img_):
    """
    将RGB图像转换为灰度图后，将灰度图的结果赋值给每一通道
    :param img_:
    :return:
    """
    img_a = np.zeros(img_.shape, dtype=np.uint8)
    gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    img_a[:, :, 0] = gray
    img_a[:, :, 1] = gray
    img_a[:, :, 2] = gray

    return img_a


# 图像旋转
def face_random_rotate(image, pts, angle, Eye_Left, Eye_Right, fix_res=False, img_size=(256, 256), vis=False):
    """
    :param image: 要处理的图像
    :param pts: 关键点信息
    :param angle: 旋转的角度
    :param Eye_Left: 左眼关键点
    :param Eye_Right: 右眼关键点
    :param fix_res: 分辨率不变
    :param img_size: 图像的大小
    :param vis: 是否显示图像
    :return:
    """
    # 获取左眼和右眼的中心点坐标
    cx, cy = (Eye_Left[0] + Eye_Right[0]) / 2, (Eye_Left[1] + Eye_Right[1]) / 2
    # 获取图像的宽高
    (h, w) = image.shape[:2]
    h = h
    w = w
    # 以两眼的中心为旋转中心，旋转角度为angle,缩放比例为1生成旋转矩阵M
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # 获取旋转角度的余弦和正弦值
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算新图像的大小
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 计算中心点的平移距离
    M[0, 2] += int(0.5 * nW) - cx
    M[1, 2] += int(0.5 * nH) - cy
    # 进行插值时使用的插值方法
    resize_model = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    # 按照生成的旋转矩阵对图像进行仿射变换，完成图像旋转
    img_rot = cv2.warpAffine(image, M, (nW, nH), flags=resize_model[random.randint(0, 4)])
    # 获取旋转之后的关键点信息
    pts_r = []
    # 遍历当前所有的关键点
    for pt in pts:
        # 获取关键点的坐标
        x = float(pt[0])
        y = float(pt[1])
        # 获取仿射变换后的关键点坐标
        x_r = (x * M[0][0] + y * M[0][1] + M[0][2])
        y_r = (x * M[1][0] + y * M[1][1] + M[1][2])
        # 将变换后的关键点坐标添加到列表pts_r中
        pts_r.append([x_r, y_r])
    # 获取当前的关键点坐标
    x = [pt[0] for pt in pts_r]
    y = [pt[1] for pt in pts_r]
    # 获取关键点区域的x,y坐标的最大值和最小值
    x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)

    # 随机裁剪
    translation_pixels = 50
    # 生成随机裁剪的左上角坐标和右下角坐标
    scaling = 0.3
    x1 += random.randint(-int(max((x2 - x1) * scaling, translation_pixels)), int((x2 - x1) * 0.15))
    y1 += random.randint(-int(max((y2 - y1) * scaling, translation_pixels)), int((y2 - y1) * 0.15))
    x2 += random.randint(-int((x2 - x1) * 0.15), int(max((x2 - x1) * scaling, translation_pixels)))
    y2 += random.randint(-int((y2 - y1) * 0.15), int(max((y2 - y1) * scaling, translation_pixels)))
    # 对超出图像区域的坐标进行clip
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, img_rot.shape[1] - 1))
    y2 = int(min(y2, img_rot.shape[0] - 1))
    # 裁剪图像获取旋转之后的人脸图像
    crop_rot = img_rot[y1:y2, x1:x2, :]
    # 初始化，存储裁剪之后的关键点
    crop_pts = []
    # 获取宽高
    width_crop = float(x2 - x1)
    height_crop = float(y2 - y1)
    # 遍历所有的关键点
    for pt in pts_r:
        # 获取x,y坐标
        x = pt[0]
        y = pt[1]
        # 获取裁剪后关键点坐标，并进行归一化
        crop_pts.append([float(x - x1) / width_crop, float(y - y1) / height_crop])

    # 随机镜像，镜像的概率是50%
    if random.random() >= 0.5:
        # 对人脸图像进行水平翻转
        crop_rot = cv2.flip(crop_rot, 1)
        # 翻转后的关键点
        crop_pts_flip = []
        for i in range(len(crop_pts)):
            # 翻转后x坐标发生变化，y坐标不变
            x = 1. - crop_pts[flip_landmarks_dict[i]][0]
            y = crop_pts[flip_landmarks_dict[i]][1]
            crop_pts_flip.append([x, y])
        # 获取关键点
        crop_pts = crop_pts_flip
    # 显示，有可视化设备
    if vis:
        # 对关键点进行计数
        idx = 0
        # 眼睛的8个关键点
        points_array_left_eye = np.zeros((1, 8, 2), dtype=np.int32)
        points_array_right_eye = np.zeros((1, 8, 2), dtype=np.int32)
        # 遍历所有的关键点
        for pt in crop_pts:
            # 获取x,y坐标
            x = int(pt[0] * width_crop)
            y = int(pt[1] * height_crop)
            # 在每一个关键点处绘制圆点
            cv2.circle(crop_rot, (int(x), int(y)), 2, (255, 0, 255), -1)
            # 眼睛的关键点，存储起来
            if 67 >= idx >= 60:
                points_array_left_eye[0, idx - 60, 0] = int(x)
                points_array_left_eye[0, idx - 60, 1] = int(y)
                cv2.circle(crop_rot, (int(x), int(y)), 2, (0, 0, 255), -1)
            elif 75 >= idx >= 68:
                points_array_right_eye[0, idx - 68, 0] = int(x)
                points_array_right_eye[0, idx - 68, 1] = int(y)
                cv2.circle(crop_rot, (int(x), int(y)), 2, (0, 255, 0), -1)
            idx += 1
        # 绘制眼睛的等高线
        cv2.drawContours(crop_rot, points_array_left_eye, -1, (0, 155, 255), thickness=2)
        cv2.drawContours(crop_rot, points_array_right_eye, -1, (0, 255, 155), thickness=2)
    # 宽高比不变时，进行填充
    if fix_res:
        crop_rot = letterbox(crop_rot, img_size=img_size[0], mean_rgb=(128, 128, 128))
    # 直接进行resize
    else:
        crop_rot = cv2.resize(crop_rot, img_size, interpolation=resize_model[random.randint(0, 4)])
    # 返回图像和对应的关键点
    return crop_rot, crop_pts

# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1 - c, b)
    return dst
