import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import torch
import cv2
from yolo.model import yolo_loss

from data_aug.data_aug import *
from data_aug.bbox_util import *

def convert_ground_truth(gt_boxes, input_shape, anchors, num_classes,batch_size):
    '''convert ground truth boxes into yolo_outputs frame as following functions:
        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        bw = pw*exp(tw)
        bh = ph*exp(th)

        Parameters
        ----------
        input_shape: model input shape, such as (416,416)
        gt_boxes: list of ground truth boxes
            [[x_min, y_min, x_max, y_max, class_id],[x_min, y_min, x_max, y_max, class_id],...]
        anchors: anchros array, shape=(9, 2)
        num_classes: .number of classes, integer
        Returns
        -------
        y_true: list of array, shape like yolo_outputs

        '''
    grid = [32,16,8]
    num_layers = len(anchors)//3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    m = len(gt_boxes)     # batch_size
    # initialize y_true with zeros
    y_true = [np.zeros((m, num_layers, num_classes + 5, input_shape[0] // grid[i], input_shape[0] // grid[i]),
                       dtype='float32') for i in range(num_layers)]
    for i in range(m):
        true_boxes = np.array(gt_boxes[i], dtype='float32')
        if len(true_boxes) !=0:
            true_boxes = np.expand_dims(true_boxes,0)
            input_shape = np.array(input_shape, dtype='int32')
            boxes_xy = (true_boxes[...,0:2]+true_boxes[...,2:4])/2
            boxes_wh = true_boxes[...,2:4] - true_boxes[...,0:2]
            anchor = np.array(anchors)
            anchor = np.expand_dims(anchor,0)
            # Find best anchor for each true box
            wh = np.expand_dims(boxes_wh, -2)
            xmin = -wh[0] / 2
            xmax = wh[0] / 2
            anchormax = anchor / 2
            anchormin = -anchormax
            intersect_mins = np.maximum(xmin, anchormin)  # use np.maximum
            intersect_max = np.minimum(xmax, anchormax)
            intersect_w = (intersect_max[..., 0] - intersect_mins[..., 0])
            intersect_h = (intersect_max[..., 1] - intersect_mins[..., 1])
            intersect_aera = intersect_w * intersect_h
            x_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchor[..., 0] * anchor[..., 1]
            iou = intersect_aera / (anchor_area + x_area - intersect_aera)
            best_anchor = np.argmax(iou, axis=-1)
            # convert ground truth boxes
            for t, n in enumerate(best_anchor.ravel()):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        xy = np.floor(boxes_xy[0, t] / grid[l]).astype('int32')
                        y_xy = boxes_xy[0, t] / grid[l] - xy
                        y_wh = np.log(boxes_wh[0, t]/anchor[0, n])
                        anchro_id = anchor_mask[l].index(n)
                        class_id = true_boxes[0, t, 4]
                        y_true[l][i, anchro_id, 0:2, xy[0], xy[1]] = y_xy
                        y_true[l][i, anchro_id, 2:4, xy[0], xy[1]] = y_wh
                        y_true[l][i, anchro_id, 4, xy[0], xy[1]] = 1
                        y_true[l][i, anchro_id, int(class_id + 5), xy[0], xy[1]] = 1
        else: continue
    return y_true

def resize(image,input_shape):
    '''
    resize image with unchanged aspect ratio using padding
    return: resized image,ratio
    '''
    img_w, img_h = image.shape[1], image.shape[0]
    w, h = input_shape
    ratio = min(w / img_w, h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    empty = np.zeros((w, h, 3), dtype='uint8')
    empty[0:new_h, 0:new_w, :] = resized_image
    return empty,ratio


def get_input_data(image):
    image_data = np.array(image,dtype='float32')
    image_data /= 255
    image_data = np.array([image_data[..., i] for i in range(3)])
    return image_data


def IOU(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    s1 = abs(bottom1 - top1) * abs(right1 - left1)
    s2 = abs(bottom2 - top2) * abs(right2 - left2)
    cross = max((min(bottom1, bottom2) - max(top1, top2)), 0) * max((min(right1, right2) - max(left1, left2)), 0)
    return cross / (s1 + s2 - cross) if (s1 + s2 - cross)!=0 else 0

def convert_yolo_outputs(out_puts, input_shape, ratio, anchors, classes, confidence = 0.05, NMS = 0.5, CUDA= True):
    '''convert yolo out puts into object boxes with following functions:
           bx = sigmoid(tx) + cx
           by = sigmoid(ty) + cy
           bw = pw*exp(tw)
           bh = ph*exp(th)

           Parameters
           ----------
           out_puts : yolo out puts
           input_shape: model input shape, such as (416,416)
           gt_boxes: list of ground truth boxes
               [[x_min, y_min, x_max, y_max, class_id],[x_min, y_min, x_max, y_max, class_id],...]
           anchors: anchros array, shape=(9, 2)
           classes: list of classes
           confidence : confidence threshold
           NMS ： NMS threshold
           Returns
           -------
           object boxes: list of boxes

           '''
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_layers = len(anchor_mask)
    num_classes = len(classes)
    input_shape = np.array(input_shape)
    out_box = []
    out_scor = []
    out_class = []
    for k in range(out_puts[0].shape[0]):
        out_boxes = []
        out_scores = []
        out_classes = []
        for i in range(num_layers):
            scal = out_puts[i].cpu().data.shape[-1]
            pred = out_puts[i].cpu().data.reshape(-1,3,num_classes+5,scal,scal)[k,...].unsqueeze(0)
            anchor = [anchors[anchor_mask[i][k]] for k in range(3)]
            anchor = torch.FloatTensor(anchor)
            grid = np.meshgrid(range(scal),range(scal))
            grid = torch.FloatTensor(grid[::-1]).unsqueeze(0).repeat(3,1,1,1).unsqueeze(0)
            # 用GPU完成张量运算
            if CUDA:
                pred = pred.cuda()
                anchor = anchor.cuda()
                grid = grid.cuda()
            # 计算预测框包含目标的置信度score
            confidence_prob =  torch.sigmoid(pred[..., 4, :, :]).unsqueeze(2)
            object_mask =  (confidence_prob > confidence).float()
            cla_prob =  torch.sigmoid(pred[..., 5:, :, :])
            scores = cla_prob * confidence_prob * object_mask
            # 计算预测框中心点坐标x,y
            x_y = (torch.sigmoid(pred[..., 0:2, :, :]) + grid) * object_mask / scal * input_shape[0] / ratio
            # 计算预测框的长宽h,w
            x = anchor[:, 0].view(-1, 1).repeat(1, scal * scal).reshape(3, scal, scal).unsqueeze(1)
            y = anchor[:, 1].view(-1, 1).repeat(1, scal * scal).reshape(3, scal, scal).unsqueeze(1)
            anchor_xy = torch.cat((x, y), 1)
            w_h = torch.exp(pred[..., 2:4, :, :]) * anchor_xy * object_mask / ratio
            # 转换成需要的输出数据格式
            scores_data = scores.cpu().data.numpy()
            x_y_data = x_y.cpu().data.numpy()
            w_h_data = w_h.cpu().data.numpy()
            object_mask = object_mask.cpu().data.numpy()
            position = np.argwhere(object_mask > 0)
            for p in position:
                b, l, c, r, cl = p
                score = np.max(scores_data[b, l, :, r, cl])
                cla_id = np.argmax(scores_data[b, l, :, r, cl])
                xy = x_y_data[b, l, :, r, cl]
                wh = w_h_data[b, l, :, r, cl]
                cla = classes[cla_id]
                xmin = max(int(xy[0] - wh[0] / 2), 0)
                xmax = min(int(xy[0] + wh[0] / 2), int(input_shape[1]/ratio))
                ymin = max(int(xy[1] - wh[1] / 2), 0)
                ymax = min(int(xy[1] + wh[1] / 2), int(input_shape[0]/ratio))
                out_boxes.append([xmin, ymin, xmax, ymax])
                out_scores.append(score)
                out_classes.append(cla)
        if NMS:
            length = len(out_scores)
            remove = []
            for i in range(length):
                for j in range(length):
                    if IOU(out_boxes[i],out_boxes[j])> NMS and out_scores[i] < out_scores[j]:
                        remove.append(i)
                        break
            for r in remove:
                out_boxes.pop(r-length)
                out_scores.pop(r-length)
                out_classes.pop(r-length)
        out_box.append(out_boxes)
        out_scor.append(out_scores)
        out_class.append(out_classes)
    return out_box,out_scor,out_class

def data_generator(annotation_lines,input_shape,anchors, num_classes,batch_size,step):
    image = []
    gt_boxes = []
    for annotation in annotation_lines[step*batch_size:(step+1)*batch_size]:
        annotation = annotation.strip()
        img = cv2.imread(annotation.split()[0])[:,:,::-1]
        boxes = np.array([np.array(box.split(',')) for box in annotation.split()[1:]],dtype=np.float32)
        img,ratio = resize(img,input_shape)
        boxes[...,:4] *=ratio
        temp_boxes = np.array([[100,100,200,200,1]],dtype = np.float32)
        transforms = Sequence([RandomHorizontalFlip(0.5), RandomRotate(10, remove=0.8),
                               RandomScale(0.2, diff=True, remove=0.8), RandomShear(0.1),
                               RandomTranslate(0.1, diff=True, remove=0.8), RandomHSV(20, 40, 40)])
        if len(boxes) !=0:
            img, boxes = transforms(img, boxes)
        else:
            img, temp_boxes = transforms(img, temp_boxes)

        image_data = get_input_data(img)
        image.append(image_data)
        gt_boxes.append(boxes)
    image = np.array(image)
    y_true = convert_ground_truth(gt_boxes,input_shape,anchors,num_classes,batch_size)
    X = torch.from_numpy(image)
    return X,y_true,ratio

def eval(model,val,input_shape,batch_size, anchors,classes,CUDA,
         optimizer = None,loss_function = None,train = False):
    if train == False:
        model.eval()
    val_lines = open(val,'r').readlines()
    if '\n' in val_lines:
        val_lines.remove('\n')
    np.random.shuffle(val_lines)
    steps = len(val_lines)//batch_size
    num_classes = len(classes)
    precision = {}
    recall = {}
    for i in classes:
        precision[i] = []
        recall[i] = []
    for step in range(steps):
        sys.stdout.write('\r')
        sys.stdout.write("evaluating validation data...%d//%d" % (int(step + 1), int(steps)))
        sys.stdout.flush()
        if train == True:
            optimizer.zero_grad()
        X, y_true,ratio = data_generator(val_lines, input_shape, anchors, num_classes, batch_size, step)
        if CUDA:
            X = X.cuda()
        out_puts = model(X)
        if train == True:
            loss = yolo_loss(out_puts, y_true, num_classes, CUDA, loss_function=loss_function,print_loss = False)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        out_box, out_score, out_class = convert_yolo_outputs(out_puts, (416, 416), ratio, anchors,
                                                                  classes, confidence=0.8, NMS=0.3, CUDA=True)
        for k, v in enumerate(val_lines[step * batch_size:(step + 1) * batch_size]):
            gt_boxes = []
            gt_classes = []
            for gt in v.strip().split(' ')[1:]:
                gt_boxes.append(list(map(int, gt.split(',')[:-1])))
                gt_classes.append(classes[int(gt.split(',')[-1].strip())])
            out_classes = out_class[k]
            out_boxes = out_box[k]
            # 计算ap
            for i in range(len(out_classes)):
                for j in range(len(gt_classes)):
                    if IOU(gt_boxes[j], out_boxes[i]) > 0.5 and gt_classes[j] == out_classes[i]:
                        precision[out_classes[i]].append(1)
                        break
                else:
                    precision[out_classes[i]].append(0)
                # 计算ar
            for i in range(len(gt_classes)):
                for j in range(len(out_classes)):
                    if IOU(gt_boxes[i], out_boxes[j]) > 0.5 and out_classes[j] == gt_classes[i]:
                        recall[gt_classes[i]].append(1)
                        break
                else:
                    recall[gt_classes[i]].append(0)
    print('\n')
    ap = []
    ar = []
    for k in precision.keys():
        p = sum(precision[k]) / len(precision[k]) if len(precision[k]) != 0 else 0
        r = sum(recall[k]) / len(recall[k]) if len(recall[k]) != 0 else 0
        ap.append(p)
        ar.append(r)
        print(k, 'AP:', '%.3f' % (p), 'AR:', '%.3f' % (r))
    print('mAP :', '%.3f' % float(sum(ap)/len(ap)), 'mAR:', '%.3f' % float(sum(ar)/len(ar)))
    return sum(ap)/len(ap),sum(ar)/len(ar)



# test
# with open('test.txt') as f:
#     lines_train = f.readlines()
# annotation_line = lines_train[0].strip()
# image = Image.open('test.jpg')
# im_copy = image.copy()
# draw = ImageDraw.Draw(im_copy)
# boxes = np.array([np.array(list(map(int, box.split(',')))) for box in annotation_line.split()[1:]])
# for box in boxes:
#     draw.rectangle(tuple(box[:4]),outline=(0,0,0), width=4)
# del draw
# plt.imshow(im_copy)
# plt.show()
# image_data, box_data =preprocess_image(annotation_line, (416,416))
# image1 = Image.fromarray((image_data*255).astype('uint8')).convert('RGB')
# # draw = ImageDraw.Draw(image1)
# # for box in box_data:
# #     draw.rectangle(tuple(list(map(int,box[:4]))),outline=(0,0,0), width=4)
# # del draw
# # plt.imshow(image1)
# # plt.show()
#
# gt_box = []
# gt_box.append(box_data)
# anchor = [[21,21], [36,36], [54,54], [75,72], [99,95], [121,134], [179,179], [268,283], [475,480]]
# y = convert_ground_truth(gt_box,(416,416),anchor,10)
# for i in range(3):
#     y[i] = torch.from_numpy(y[i])
# classes = [0,1,2,3,4,5,6,7,8,9]
# out_boxes,out_scores,out_classes =convert_yolo_outputs(y,(416,416),1,
#                                                        anchor, classes,confidence = 0.05,NMS = 0.5,CUDA= True)
#
# draw = ImageDraw.Draw(image1)
# boxes = out_boxes
# for box in boxes:
#     draw.rectangle(tuple(box[:4]),outline=(0,0,0), width=4)
# del draw
# plt.imshow(image1)
# plt.show()

# for i in range(3):
#     z = np.array(y[i])
#     for j in range(3):
#         for k in range(z.shape[2]):
#             for n in range(z.shape[0]):
#                 q = z[n,j,k,:,:]
#                 if q.sum()!= 0:
#                     # plt.imshow(q)
#                     # plt.title('-'.join(list(map(str,[n,i,j,k])))+'  '+'%3.3f'%(q.sum()))
#                     # plt.show()
#                     print(n,i,j,k,'%3.3f'%(q.sum()))








