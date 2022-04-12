# 指定区间
# lr_scheduler.MultiStepLR()
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
# lr = 0.0005   if epoch >= 80
import torch.optim as optim
import matplotlib.pyplot as plt
import yoloV3
import torch

model = yoloV3.Yolov3(1)
optimizer = optim.SGD(params=model.parameters(), lr=0.05)

plt.figure()
y=[]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 80], 0.1)
for epoch in range(100):
    scheduler.step()
    print(epoch, 'lr={:.6f}'.format(scheduler.get_lr()[0]))
    y.append(scheduler.get_lr()[0])

plt.plot(y)
plt.show()
