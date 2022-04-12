from torch.utils.data import DataLoader
import os
import cv2
import numpy as np

# 指定文件路径
# 数据路径
root_path = '/root/cv/dataset/人脸/datasets'
# txt文件的路径
path = '/root/cv/dataset/人脸/datasets/yolo_widerface_open_train/anno/train.txt'
# 要检测的类别
path_voc_names = './cfg/face.names'

if __name__ == '__main__':
    # 读取检测的目标类别
    with open(path_voc_names, 'r') as f:
        label_map = f.readlines()
    # 获取所有的类别
    for i in range(len(label_map)):
        label_map[i] = label_map[i].strip()
        print(i, ') ', label_map[i].strip())
    # 获取所有的图像文件
    with open(path, 'r') as file:
        img_files = file.read().splitlines()
        img_files = list(filter(lambda x: len(x) > 0, img_files))

    for i in range(len(img_files)):
        img_files[i] = img_files[i].replace('./', '/root/cv/dataset/人脸/datasets/')
    # 获取所有的标注文件
    label_files = [
        x.replace('images', 'labels').replace('.jpg', '.txt').replace('./', '/root/cv/dataset/人脸/')
        for x in img_files]
    # 读取图像并对标注信息进行绘制
    # for i in range(len(img_files)):
    for i in range(100):
        # print(img_files[i])
        # 图像的绝对路径
        # print(img_files[i][2:])
        img_file = os.path.join(img_files[i])
        # 图像读取，获取宽高
        # print(img_file)
        img = cv2.imread(img_file)
        w = img.shape[1]
        h = img.shape[0]
        # 标签文件的绝对路径
        label_path = os.path.join(label_files[i])
        # print(i, label_path)
        if os.path.isfile(label_path):
            # 获取每一行的标注信息
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()
            # 获取每一行中的标准信息(cls,x,y,w,h)
            x = np.array([x.split() for x in lines], dtype=np.float32)
            for k in range(len(x)):
                anno = x[k]
                label = int(anno[0])
                # 获取框的坐标值，左上角坐标和右下角坐标
                x1 = int((float(anno[1]) - float(anno[3]) / 2) * w)
                y1 = int((float(anno[2]) - float(anno[4]) / 2) * h)

                x2 = int((float(anno[1]) + float(anno[3]) / 2) * w)
                y2 = int((float(anno[2]) + float(anno[4]) / 2) * h)

                # 将标注框绘制在图像上
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 30, 30), 2)
                # 将标注类别绘制在图像上
                cv2.putText(img, ("%s" % (str(label_map[label]))), (x1, y1), \
                            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 55), 6)
                cv2.putText(img, ("%s" % (str(label_map[label]))), (x1, y1), \
                            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 55, 255), 2)
                cv2.imwrite("./samples/results_{}".format(os.path.basename(img_file)), img)

    # 结果显示
    # cv2.namedWindow('image', 0)
    # cv2.imshow('image', img)
    # if cv2.waitKey(1) == 27:
    #     break
    # print("./samples/results_{}".format(os.path.basename(img_file)))
    # print(img_file, '---------------')
    # print("./samples/results_{}".format(os.path.basename(img_file)))
    # cv2.destroyAllWindows()
