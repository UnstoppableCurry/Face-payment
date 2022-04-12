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
使用U版就行没什么好说的
个人推荐连接 自带教程：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/yolov3_spp

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
![image](https://user-images.githubusercontent.com/65523997/162980530-5b0cb893-3522-4844-9219-daaf57fb747a.png)



![image](https://user-images.githubusercontent.com/65523997/162979445-10280313-8778-4775-bd1c-476f27eacd72.png)

这里是多任务所使用的的loss函数

![image](https://user-images.githubusercontent.com/65523997/162979465-95743e5f-85fd-4f4f-8c3c-3e99d248a6d5.png)

当∣ x ∣ < w 时，主体是一个对数函数，另外还有两个控制参数，分别是w和ε，其他取值时是一个L1损失。要找合适的 w 和 ϵ 值，要进行调参，推荐的参数如下表所示：


![image](https://user-images.githubusercontent.com/65523997/162979534-1e56019d-883b-41a9-bc2a-6acf2d9b673a.png)

数据集的名称貌似是 wiki_crop_face_multi_task 格式如下

![image](https://user-images.githubusercontent.com/65523997/162979733-2174fc9d-91c8-4d66-9b38-b30ada36faf6.png)



 
数据增强的逻辑

送入网络中图像如下图所示：（第一排是原始图像，第二排是送入网络中的图像）

 

![image](https://user-images.githubusercontent.com/65523997/162980576-b18ccc39-53e7-4347-9b18-153129031a8e.png)



架构很简单

![image](https://user-images.githubusercontent.com/65523997/162980835-a9b1db8d-54b8-4b5e-9332-ce9800c52856.png)

三任务联合训练时的loss

![image](https://user-images.githubusercontent.com/65523997/162980926-722dbaa6-2b4f-4bf5-b7e0-40b3bd3f9ebe.png)

如果将姿态任务添加进去呢？
暴力添加 这里简单说一下，并没有很全的数据 所以直接teacherforcing思想 用训练好的姿态网络作为老师以此计算loss

![loss (2)](https://user-images.githubusercontent.com/65523997/162981190-9de7c23a-9d02-40c1-ada9-de8666c1bc5d.png)

 评估时效果很差 再次调整 将姿态任务的loss权重修改后
![loss](https://user-images.githubusercontent.com/65523997/162981514-e60c1818-7cc7-4b05-a4fa-d035f79b4038.png)

这里有不小的坑 反正自己比较菜即使这样效果也不理想

多任务训练时, 步长一定时, batch越大拟合的越快,但是如果开启数据增强,则拟合的更慢
Batch越小,所需步长越大 但是问题来了

![image](https://user-images.githubusercontent.com/65523997/162981826-cdd4f3db-d688-4355-8528-3f8190a48ae5.png)

如果按照训练相对时间来看, batchsize 未必越大越好, 相同时间的情况下,batch 越大拟合的未必越快

![image](https://user-images.githubusercontent.com/65523997/162981893-b488438a-1764-4f4c-8dfe-38e525aebccd.png)

第三张图是学习量相同时的拟合速度,可以见到拟合最快的是batchsize64所以可见batchsize未必越大越好,而是合适就行

![image](https://user-images.githubusercontent.com/65523997/162982140-a36a9447-9959-4b50-8767-c8a6a575fbe4.png)

效果图

![image](https://user-images.githubusercontent.com/65523997/162982342-3b14c017-0615-43a4-a30b-27cb6f0cc2f9.png)

# 姿态检测
人脸姿态估计指的是根据一幅二维的人脸图像，计算出其在实际三维空间中的面部朝向。输入就是一张二维人脸图片，输出表示方位的三个旋转角度 (pitch, yaw, roll)（欧拉角），其中 pitch 表示俯仰角（关于x轴的旋转角度），yaw 表示偏航角（关于y轴的旋转角度），roll 表示翻滚角（关于z轴的旋转角度），分别对应则这抬头，摇头和转头，如下图所示（我们把人脸理解为一架向我们飞来的飞机）
![image](https://user-images.githubusercontent.com/65523997/162982455-5571f626-7c37-4cf7-ac6f-d48647285819.png)


数据集貌似叫这个  face_euler_angle_datasets  我是发不了 大家理解
格式是这样的

![image](https://user-images.githubusercontent.com/65523997/162982720-462ceb7d-5f17-48bb-8971-4dd5a7911c88.png)

训练也很简单 就是直接回归欧拉角

![image](https://user-images.githubusercontent.com/65523997/162982852-1e4ffbca-69f9-46a6-a1ec-3763f04289d0.png)



