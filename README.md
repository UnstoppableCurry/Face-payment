# Face-payment
源码实现提交
# 本项目是人脸支付级别的 人脸项目 源码实现仅供交流学习使用 不可商用
项目架构
- 1.人脸检测实现目标框锁定
- 2.人脸108关键点检测 实现 人脸矫正 年龄预测 性别预测   还可实现活体检测（例如计算人眼关键点横纵比变换）
- 3.人脸姿态识别  欧拉角不符合预期时不予进行身份校验
- 4.人脸识别 实现一对多 十万分之一误检率 通过准确率50%

# 代码细节与跑的坑
-1.首先声明 这个项目不可开源商用 所以不方便为大家提供已经训练好的任何模型与数据，并且也不提供任何数据集 因为涉及个人隐私 非隐私数据可自行寻找开源数据集

-2.第一个坑 提供的yoloV3 代码如果说自行训练，90%的概率梯度不下降，因为不是标准U版代码实现 没有使用梯度累计 数据增强 学习率退火等优化方法 ，
如果你的显卡设备显存不够大 请不要使用此代码，batchsize太小了训练不出来啊

-3.第二个坑 不要试图用本项目提供的yolov3-tiny代码尝试训练。。。 预测可以 但是NMS的实现貌似也跟U版有很大问题，自己联调的时候直接修改U版V3-SPP的NMS 
如果有需求要同时检测多人就别改，不需要 仅检测一个人的数据则topk取1就行 当然了NMS还有其他两个阈值 也是要看个人喜好进行更改

![image](https://user-images.githubusercontent.com/65523997/162975652-47f50c12-f0ad-44b1-868d-b9dd12098bf1.png)
 
-4.第三个坑 关键点检测与姿态识别的backbone都是使用resnet ，但是96关键点的输出头是三个多任务（关键点-年龄-性别）使用winloss联合训练，个人尝试过将姿态也
加入winloss中进行进行联合训练，但是精度并不理想代码实现中也有
 
 数据集说明：公开的数据集比较少超过68个关键点，其中比较有名的是Wider Facial Landmark in the Wild（WFLW），它提供了98个关键点。 WFLW 包含了 10000 张脸，其中 7500 用于训练，2500 张用于测试。除了关键点之外，还有遮挡、姿态、妆容、光照、模糊和表情等信息的标注。
 
# 项目讲解
# yoloV3
使用U版就行没什么好说的 个人推荐连接 自带教程：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp
数据集大家找找应该可以找到的 大概名称是yolo_widerface_open_train 我也不好乱发出来
网络结构
 ![image](https://user-images.githubusercontent.com/65523997/162977659-c5081bb3-c4c1-46e4-b015-59ae34423b59.png)
![image](https://user-images.githubusercontent.com/65523997/162977702-f5380ed5-6cb5-479d-81cf-dc040013091f.png)
学习率预热与退火策略可视化
![image](https://user-images.githubusercontent.com/65523997/162977856-5110b03f-4b8f-4f17-ae8c-a97f42d5829c.png)

loss函数
![image](https://user-images.githubusercontent.com/65523997/162977917-0da9431a-bafd-4a68-aa24-84e445261d35.png)

这里说一下，如果说训练时难以拟合的时候 建议在yolo的训练代码里将 anchors的权重适当增加，或者说那个部分的loss降不下去就将对应的公式权重增加
另外梯度累计与数据增强大家也要看一下 不多废话想必学习这个项目的人都有这个基础了

# 人脸108关键点检测 多任务

96关键点检测多任务实现
系统架构

![image](https://user-images.githubusercontent.com/65523997/162979445-10280313-8778-4775-bd1c-476f27eacd72.png)

这里是多任务所使用的的loss函数

![image](https://user-images.githubusercontent.com/65523997/162979465-95743e5f-85fd-4f4f-8c3c-3e99d248a6d5.png)

![image](https://user-images.githubusercontent.com/65523997/162979534-1e56019d-883b-41a9-bc2a-6acf2d9b673a.png)

当∣ x ∣ < w 时，主体是一个对数函数，另外还有两个控制参数，分别是w和ε，其他取值时是一个L1损失。要找合适的 w 和 ϵ 值，要进行调参，推荐的参数如下表所示：
![image](https://user-images.githubusercontent.com/65523997/162979733-2174fc9d-91c8-4d66-9b38-b30ada36faf6.png)

数据集的名称貌似是 wiki_crop_face_multi_task 格式如下

 
数据增强的逻辑

送入网络中图像如下图所示：（第一排是原始图像，第二排是送入网络中的图像）

![Uploading image.png…]()



