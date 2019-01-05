#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/27 9:25
# @Site    : 
# @File    : train.py
# @Software: PyCharm
import torch
import torch.nn.functional as F
import numpy as np
from utiliy import data_generator
from model import yolo_body,yolo_loss


def train(yolo_model,feeeze_body,epoche,batch_size,annotations,val, input_shape, anchors, num_classes,CUDA):
    annotation_lines = open(annotations,'r').readlines()
    steps = len(annotation_lines)//batch_size
    if feeeze_body:
        optimizer = torch.optim.Adam([{'params': yolo_model.yolo_block1.conv7.parameters()},
                                      {'params': yolo_model.yolo_block2.conv7.parameters()},
                                      {'params': yolo_model.yolo_block3.conv7.parameters()}], lr=1e-5)
    else:
        optimizer = torch.optim.Adam(yolo_model.parameters(),lr=1e-5)
    val_lines = open(val,'r').readlines()
    if CUDA:
        yolo_model.cuda()
    for i in range(epoche):
        print('train yolov3 on epoch {} with bath size {}'.format(i+1,batch_size))
        np.random.shuffle(annotation_lines)
        for step in range(steps):
            print('step: {}/{}'.format(step+1,steps))
            X, y_true = data_generator(annotation_lines, input_shape, anchors, num_classes, batch_size, step)
            if CUDA:
                X = X.cuda()
            optimizer.zero_grad()
            Y = yolo_model(X)
            loss = yolo_loss(Y,y_true,num_classes,CUDA)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

if __name__ == '__main__':
    CUDA = True
    anchors = open('yolo_anchors.txt').read().strip()
    anchors = np.array([float(b) for b in anchors.split(',')])
    anchors = anchors.reshape(-1,2)
    input_shape = (256,256)
    classes = open('train_data\\train_classes.txt').readlines()
    classes = [c.strip() for c in classes]
    num_classes = len(classes)
    annotations = 'data_train.txt'
    val = 'data_val.txt'
    weight_path = 'yolov3_state_dict.pt'
    batch_size = 2
    epoche = 50
    print('load model...')
    yolo = yolo_body(num_classes)
    yolo.load_weight(weight_path)
    print('load {} successed!'.format(weight_path))
    feeeze_body = True
    train(yolo, feeeze_body, epoche, batch_size, annotations, val, input_shape, anchors, num_classes,CUDA)
    torch.save('aa.pt',yolo.state_dict())
