path = '/www/dataset/yolo_helmet_train/yolo_helmet_train/anno/train.txt'

with open(path, 'r') as file:
    img_files = file.read().splitlines()
    print(img_files)
    img_files = list(filter(lambda x: len(x) > 0, img_files))