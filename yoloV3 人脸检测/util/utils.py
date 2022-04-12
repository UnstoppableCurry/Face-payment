import glob
import random
import time
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
import os
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        torch.manual_seed_all(seed)


def load_classes(path):
    # Loads class labels at 'path'
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %60s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        # name = name.replace('module_list.', '')
        print('%5g %60s %9s %12g %20s %10.3g %10.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (i + 1, n_p, n_g))


# 在图像上绘制检测框
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # 线宽
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    # 颜色
    color = color or [random.randint(0, 255) for _ in range(3)]
    # 左上角坐标和右下角坐标
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # 绘制矩形
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    # 若label不为空
    if label:
        # 线宽
        tf = max(tl - 1, 1)
        # 字体大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # 文字区域的右下角坐标
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # 绘制框底色
        cv2.rectangle(img, c1, c2, color, -1)
        # 添加文字
        cv2.putText(img, label, (c2[1]+100, c2[0]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# image size(416*416) 转为原图尺寸
def scale_coords(img_size, coords, img0_shape):
    # gain  = old / new 相对于原图的变换比例
    gain = float(img_size) / max(img0_shape)
    # 获取宽高的padding的大小
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    # 将坐标值去除padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    # 缩放
    coords[:, :4] /= gain
    # 坐标的最小值不为负数
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            # plt.plot(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    return inter_area / union_area  # iou


def wh_iou(box1, box2):
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets,nt,bs):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
    lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])  # losses 初始化 为 0
    txy, twh, tcls, indices = targets
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()  # 多标签分类时 使用 如 [1,1,0], sigmoid+多分类损失

    # Compute losses
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image_idx, anchor_idx, gridx, gridy

        # print(i,') b, a, gj, gi : ')
        # print('b', b)
        # print('a', a)
        # print('gj', gj)
        # print('gi', gi)

        tconf = torch.zeros_like(pi0[..., 0])  # conf

        # print('tconf: ',tconf.size())
        # Compute losses
        # k = 1  # nT / bs
        k = nt/(bs*(len(str(nt))-1))  # nT / bs
        if len(b) > 0:
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            tconf[b, a, gj, gi] = 1  # conf

            lxy += (k * 16) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
            lwh += (k * 4) * MSE(pi[..., 2:4], twh[i])  # wh loss
            lcls += (k * 1) * CE(pi[..., 5:], tcls[i])  # class_conf loss

        lconf += (k * 64) * BCE(pi0[..., 4], tconf)  # obj_conf loss
    loss = lxy + lwh + lconf + lcls

    # Add to dictionary
    d = defaultdict(float)
    losses = [loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item()]
    for name, x in zip(['total', 'xy', 'wh', 'conf', 'cls'], losses):
        d[name] = x

    return loss, d


def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    txy, twh, tcls, indices = [], [], [], []
    for i, layer in enumerate(get_yolo_layers(model)):  # 遍历 3 个 yolo layer
        # print(i,'layer ',model.module_list[layer])
        layer = model.module_list[layer][0]

        # iou of targets-anchors
        gwh = targets[:, 4:6] * layer.ng  # 以 grid 为单位的 wh
        iou = [wh_iou(x, gwh) for x in layer.anchor_vec]
        iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

        # reject below threshold ious (OPTIONAL, increases P, lowers R)
        reject = True
        if reject:
            j = iou > 0.10
            t, a, gwh = targets[j], a[j], gwh[j]
        else:
            t = targets

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * layer.ng
        gi, gj = gxy.long().t()  # grid_i, grid_j
        indices.append((b, a, gj, gi))  # img_index , anchor_index , grid_x , grid_y

        # print('b, a, gj, gi : ')
        # print('b', b)
        # print('a', a)
        # print('gj', gj)
        # print('gi', gi)
        # print('class c',c)

        # XY coordinates
        txy.append(gxy - gxy.floor())  # 转化为grid相对坐标

        # Width and height
        twh.append(torch.log(gwh / layer.anchor_vec[a]))  # yolo method 对数
        # twh.append(torch.sqrt(gwh / layer.anchor_vec[a]) / 2)  # power method

        # Class
        tcls.append(c)
        # try:
        #     print('c.max,layer.nC: ',c.max().item() ,layer.nC)
        # except:
        #     pass
        if c.shape[0]:
            assert c.max().item() <= layer.nc, 'Target classes exceed model classes'

    return txy, twh, tcls, indices


# @profile
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.6):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_conf, class)
    """

    min_wh = 2  # (pixels) minimum box width and height

    output = [None] * len(prediction)
    for image_i, pred in enumerate(prediction):
        # Experiment: Prior class size rejection
        # x, y, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        # a = w * h  # area
        # ar = w / (h + 1e-16)  # aspect ratio
        # n = len(w)
        # log_w, log_h, log_a, log_ar = torch.log(w), torch.log(h), torch.log(a), torch.log(ar)
        # shape_likelihood = np.zeros((n, 60), dtype=np.float32)
        # x = np.concatenate((log_w.reshape(-1, 1), log_h.reshape(-1, 1)), 1)
        # from scipy.stats import multivariate_normal
        # for c in range(60):
        # shape_likelihood[:, c] =
        #   multivariate_normal.pdf(x, mean=mat['class_mu'][c, :2], cov=mat['class_cov'][c, :2, :2])

        # Filter out confidence scores below threshold
        class_conf, class_pred = pred[:, 5:].max(1)  # max class_conf, index
        pred[:, 4] *= class_conf  # finall conf = obj_conf * class_conf
        i = (pred[:, 4] > conf_thres) & (pred[:, 2] > min_wh) & (pred[:, 3] > min_wh)
        c = (pred[:, 4] > conf_thres)  # [x for x in torch.sort(pred[:, 4], 0,descending=True)]
        # xx = 0
        # for i in c:
        #     if i:
        #         xx += 1

        # s2=time.time()
        # pred2 = pred[i]
        # print("++++++pred2 = pred[i]",time.time()-s2, pred2)
        pred2 = pred[i]
        # If none are remaining => process next image
        if len(pred2) == 0:
            continue

        # Select predicted classes
        class_conf = class_conf[i]
        class_pred = class_pred[i].unsqueeze(1).float()

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        pred2[:, :4] = xywh2xyxy(pred2[:, :4])
        # pred[:, 4] *= class_conf  # improves mAP from 0.549 to 0.551

        # Detections ordered as (x1y1x2y2, obj_conf, class_conf, class_pred)
        pred2 = torch.cat((pred2[:, :5], class_conf.unsqueeze(1), class_pred), 1)

        # Get detections sorted by decreasing confidence scores
        pred2 = pred2[(-pred2[:, 4]).argsort()]

        det_max = []
        nms_style = 'MERGE'  # 'OR' (default), 'AND', 'MERGE' (experimental)
        for c in pred2[:, -1].unique():
            dc = pred2[pred2[:, -1] == c]  # select class c
            dc = dc[:min(len(dc), 100)]  # limit to first 100 boxes

            # Non-maximum suppression
            if nms_style == 'OR':  # default
                # METHOD1
                # ind = list(range(len(dc)))
                # while len(ind):
                # j = ind[0]
                # det_max.append(dc[j:j + 1])  # save highest conf detection
                # reject = (bbox_iou(dc[j], dc[ind]) > nms_thres).nonzero()
                # [ind.pop(i) for i in reversed(reject)]

                # METHOD2
                while dc.shape[0]:
                    det_max.append(dc[:1])  # save highest conf detection
                    if len(dc) == 1:  # Stop if we're at the last detection
                        break
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'AND':  # requires overlap, single boxes erased
                while len(dc) > 1:
                    iou = bbox_iou(dc[0], dc[1:])  # iou with other boxes
                    if iou.max() > 0.5:
                        det_max.append(dc[:1])
                    dc = dc[1:][iou < nms_thres]  # remove ious > threshold

            elif nms_style == 'MERGE':  # weighted mixture box
                while len(dc):
                    i = bbox_iou(dc[0], dc) > nms_thres  # iou with other boxes
                    weights = dc[i, 4:5]
                    dc[0, :4] = (weights * dc[i, :4]).sum(0) / weights.sum()
                    det_max.append(dc[:1])
                    dc = dc[i == 0]

        if len(det_max):
            det_max = torch.cat(det_max)  # concatenate
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]  # sort
    return output


def get_yolo_layers(model):
    yolo_layer_index = []
    # print(model.module_list)
    for index, l in enumerate(model.module_list):
        try:
            a = l[0].img_size and l[0].ng  # only yolo layer need img_size and nG
            # print("---"*50)
            # print(l, index)
            yolo_layer_index.append(index)
        except:
            pass
    assert len(yolo_layer_index) > 0, "can not find yolo layer"
    return yolo_layer_index


def letterbox(img, height=416, augment=False, color=(127.5, 127.5, 127.5)):
    """
    保持图片的长宽比例，剩下的部分采用灰色填充，获取一个输入到网络中的正方形图片
    :param img: 要处理的图像
    :param height: 正方形图像的高
    :param augment: 是否进行图像增强
    :param color: pad部分使用其进行填充
    :return:
    """
    # shape = [height, width]，要处理图像的高宽
    shape = img.shape[:2]
    # ratio  = old / new 缩放的比例
    ratio = float(height) / max(shape)
    # resize后图像的大小
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    # 宽pad的大小
    dw = (height - new_shape[0]) / 2
    # 高pad的大小
    dh = (height - new_shape[1]) / 2
    # 获取周围pad的大小
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # 是否增强
    if augment:
        # 随机选择插值方式
        interpolation = np.random.choice([None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          None, cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                                          cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        # 若为None按默认方式增强，否则按随机选择的方式进行增强
        if interpolation is None:
            img = cv2.resize(img, new_shape)
        else:
            img = cv2.resize(img, new_shape, interpolation=interpolation)
    else:
        # 若不增强，按近邻方式进行增强
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_NEAREST)
    # 将padding的填充在图像的周围，获取方形的结果
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    # 返回图像，比例，pading大小
    return img, ratio, dw, dh


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    """

    :param img:
    :param targets:
    :param degrees: 旋转角度
    :param translate: 水平或者垂直移动的范围
    :param scale: 放缩尺度因子
    :param shear: 错切
    :param borderValue: 填充空白区域的颜色
    :return:
    """
    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets
