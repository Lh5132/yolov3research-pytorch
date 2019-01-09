import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo import model
import time
import cv2
import numpy as np
from yolo.utility import convert_yolo_outputs,convert_ground_truth,resize,get_input_data
import matplotlib.pyplot as plt

# torch.cuda.empty_cache()
# start1 = time.time()
# yolo = model.yolo_body(80)
# yolo.load_weight('yolov3_state_dict.pt')
# yolo.eval()
# yolo.cuda()
# print('load time:',time.time()-start1)
# start2 = time.time()
# image = cv2.imread('messi.jpg')
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# resize_image,ratio = resize(image,(416,416))
# image_data = get_input_data(resize_image)
# image_data = np.expand_dims(image_data,0)
# X =torch.from_numpy(image_data)
# X = X.cuda()
# with torch.no_grad():
#     out_puts = yolo(X)
# print('processing time:',time.time()-start2)
# start3 = time.time()
# #输出转换
# with open('name.txt','r') as f:
#     classes = f.readlines()
# classes = [c.strip() for c in classes]
# anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
# out_boxes,out_scores,out_classes = convert_yolo_outputs(out_puts,(416,416),ratio, anchors,
#                                                         classes,confidence = 0.8,NMS = 0.3,CUDA= True)
# for i,box in enumerate(out_boxes):
#     label = out_classes[i]
#     cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,0,0),3)
#     cv2.putText(image, label, (box[0],box[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, [225,0,0], 2)
# plt.imshow(image)
# plt.show()
# plt.imsave('test.jpg',image)
# print('Output Processing:',time.time()-start3)
# print('total time:',time.time()-start1)
# with SummaryWriter() as w:
#     w.add_graph(yolo,X,verbose=True)
from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
#
#
with open('test1.txt') as f:
    lines_train = f.readlines()
for annotation_line in lines_train:
    annotation_line = annotation_line.strip()
    bboxes = np.array([np.array(box.split(',')) for box in annotation_line.split()[1:]],dtype=np.float32)
    np.random.shuffle(bboxes)
    img = cv2.imread(annotation_line.split()[0])[:,:,::-1] #convert to RGB
    img,ratio = resize(img,(416,416))
    bboxes[...,:4] *=ratio

    transforms = Sequence([RandomHorizontalFlip(0.5), RandomRotate(10,remove=0.8),
                           RandomScale(0.2,diff=True,remove=0.8),RandomShear(0.1),
                           RandomTranslate(0.1,diff=True,remove=0.8),RandomHSV(20,40,40)])


    temp_boxes = np.array([[100,100,200,200,1]],dtype=np.float32)
    if len(bboxes) != 0:
        img, bboxes = transforms(img, bboxes)
        plt.imshow(draw_rect(img, bboxes))
        plt.show()
    else:
        img, temp_boxes = transforms(img, temp_boxes)
        plt.imshow(img)
        plt.show()
    print(bboxes)
# b = []
# b.append(trans_bboxes)
# anchor = [[21,21], [36,36], [54,54], [75,72], [99,95], [121,134], [179,179], [268,283], [475,480]]
# y = convert_ground_truth(b,(416,416),anchor,30)
