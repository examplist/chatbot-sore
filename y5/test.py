import cv2
import torch
from PIL import Image
import time

# Model
# path = r'C:/Users/kjs/Desktop/yolov5-master/runs/train/exp5/weights/'
# model = torch.hub.load('C:/Users/kjs/Desktop/yolov5-master', 'custom', path=path+'/best.pt', source='local')
model = torch.hub.load('./', 'custom', path='./runs/best_burn.pt', source='local')
model2 = torch.hub.load('./', 'custom', path='./runs/best_bedsore.pt', source='local')
# model = torch.hub.load(path, 'best')

# Images
img_path = r'./datasets/test/'
# img_path = r'./datasets/BURN512/images/Test/'

img1 = Image.open(img_path + 'b1.jpg')  # PIL image
# img2 = cv2.imread(img_path + 'b3.jpg')
# imgs = [img1, img2]  # batch of images
imgs = [img1, ] 

start = time.perf_counter_ns()

# Inference
results = model(imgs, size=512)  # includes NMS
results2 = model2(imgs, size=224)  # includes NMS

duration = (time.perf_counter_ns() - start)
print(f"검출 추론 과정 : {duration // 1000000}ms.")

# # Results
# results.print()
# results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
print(results.pandas().xyxy[0])  # img1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

results2.xyxy[0]  # img1 predictions (tensor)
print(results2.pandas().xyxy[0])  # img1 predictions (pandas)

