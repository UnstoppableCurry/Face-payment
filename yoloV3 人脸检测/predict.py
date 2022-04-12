import argparse
import time
import os
import torch
from util.datasets import *
from util.utils import *
from util.parse_config import parse_data_cfg
from yoloV3 import Yolov3, Yolov3Tiny
from util.torch_utils import select_device


# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# 图像预处理
def process_data(img, img_size=416):
    img, _, _, _ = letterbox(img, height=img_size)
    # 通道转换 BGR to RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    # 类型转换 uint8 to float32
    img = np.ascontiguousarray(img, dtype=np.float32)
    # 归一化 0 - 255 to 0.0 - 1.0
    img /= 255.0
    return img


def detect(model_path, cfg, data_cfg, img_size=416, conf_thres=0.1, nms_thres=0.2, video_path=0):
    """

    :param model_path: 模型路径
    :param cfg: 配置信息
    :param data_cfg: 数据配置信息
    :param img_size: 图像的大小
    :param conf_thres: 置信度阈值
    :param nms_thres: NMS阈值
    :param video_path: 要处理的视频路径
    :return:
    """
    # 获取检测的类别信息
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    num_classes = len(classes)
    # 1.模型加载
    # 第一步：指定训练好的模型参数
    if "-tiny" in cfg:
        model = Yolov3Tiny(num_classes)
    else:
        model = Yolov3(num_classes)
    # 第二步：加载模型训练结果
    device = select_device()
    weights = model_path
    if os.access(weights, os.F_OK):
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        return False
    use_cuda = torch.cuda.is_available()
    model.to(device).eval()

    # 2.数据加载
    video_capture = cv2.VideoCapture(video_path)

    # 3.遍历帧图像进行处理
    # 第一步：设置结果保存位置
    video_writer = None
    save_video_path = "result_{}.MP4".format(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    # 第二步：遍历视频中每一帧图像
    while (video_capture.isOpened()):
        ret, im0 = video_capture.read()
        # 第三步：图像预处理，并记录处理时间
        if ret:

            t = time.time()
            img = process_data(im0, img_size)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.time()
            print("process time", t1 - t)
            # 第四步：模型前向推理进行检测，并记录推理时间
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            pred, _ = model(img)
            if use_cuda:
                torch.cuda.synchronize()
            t2 = time.time()
            print("inference time", t2 - t1)
            # 第五步：非极大值抑制，NMS，并记录推理时间
            detections = non_max_suppression(pred, conf_thres, nms_thres)[0]
            if use_cuda:
                    torch.cuda.synchronize()
            t3 = time.time()
            print("get res time", t3 - t2)
            if detections is None or len(detections) == 0:
                cv2.imshow("image", im0)
                cv2.waitKey(1)
                print('跳过-----------------------------------------')
                continue
            # 第六步：结果展示
            detections[:, :4] = scale_coords(img_size, detections[:, :4], im0.shape).round()

            for *xyxy, conf, cls_conf, cls in detections:
                label = "%s %.2f" % (classes[int(cls)], conf)
                xyxy = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                im0 = plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=3)
            s2 = time.time()
            print("detect time", s2 - t)
            cv2.imshow("image", im0)
            cv2.waitKey(1)
            # if video_writer is None:
            #     # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #     video_writer = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps=25, framesize=(im0.shape[0], im0.shape[1]))
            # video_writer.write(im0)
    cv2.destroyAllWindows()
    video_capture.release()


# 4.模型使用
if __name__ == '__main__':
    # 配置信息设置
    # 模型相关配置文件
    data_config = './cfg/face.data'
    # 检测模型路径

    # model_path = '/root/cv/pycharm/人脸检测/yolo3-人脸检测/weights-yolov3-faceyolov3_416_epoch_9.pt'
    # model_path = 'weights-yolov3-face/yolov3_demo_416_epoch_7.pt'
    model_path = 'weights-yolov3-face/yolov3_demo_416_epoch_2.pt'
    model_cfg = 'yolo'
    # video_path = "/root/cv/pycharm/人脸检测/yolo3-人脸检测/激励自己.mp4"  # 测试视频  # 测试视频
    video_path = "./11.mp4"  # 测试视频  # 测试视频
    # 图像尺寸
    img_size = 416
    # 检测置信度
    conf_thres = 0.2
    # nms 阈值
    nms_thres = 0.6
    with torch.no_grad():
        detect(
            model_path=model_path,
            cfg=model_cfg,
            data_cfg=data_config,
            img_size=img_size,
            conf_thres=conf_thres,
            nms_thres=nms_thres,
            video_path=video_path
        )
