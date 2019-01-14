import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义C-B-L卷积三明治
class Conv2D_BN_Leaky(nn.Module):
    def __init__(self, inc, outc, kernel=3, strid=1, padding=1):
        super(Conv2D_BN_Leaky, self).__init__()
        if kernel == 1 or strid == 2:
            padding = 0
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel, stride=strid, padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.laeky = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.laeky(out)
        return out


# 定义卷积运算
class Conv2D(nn.Module):
    def __init__(self, inc, outc, kernel=3, strid=1, padding=1):
        super(Conv2D, self).__init__()
        if kernel == 1 or strid == 2:
            padding = 0
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel, stride=strid, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        return out


# 残差连接部分
class ConvolutionBlock(nn.Module):
    def __init__(self, num_filters):
        super(ConvolutionBlock, self).__init__()
        filters = num_filters
        self.conv1 = Conv2D_BN_Leaky(filters, filters // 2, kernel=1)
        self.conv2 = Conv2D_BN_Leaky(filters // 2, filters, kernel=3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out


# 定义Darknet残差结构
class resblock(nn.Module):
    def __init__(self, inc, num_filters, num_blocks):
        super(resblock, self).__init__()
        self.conv1 = Conv2D_BN_Leaky(inc, num_filters, kernel=3, strid=2)
        self.res_body = self.makelayers(num_filters, num_blocks)

    def makelayers(self, filters, blocks):
        layers = []
        for i in range(blocks):
            layers.append(ConvolutionBlock(filters))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, (1, 0, 1, 0))
        x = self.conv1(x)
        x = self.res_body(x)
        return x


# 定义Darknet
class darknet_body(nn.Module):
    def __init__(self):
        super(darknet_body, self).__init__()
        self.conv = Conv2D_BN_Leaky(3, 32)
        self.reblck1 = resblock(32, 64, 1)
        self.reblck2 = resblock(64, 128, 2)
        self.reblck3 = resblock(128, 256, 8)
        self.reblck4 = resblock(256, 512, 8)
        self.reblck5 = resblock(512, 1024, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.reblck1(x)
        x = self.reblck2(x)
        x = self.reblck3(x)
        x = self.reblck4(x)
        x = self.reblck5(x)
        return x

#yolo输出层
class yolo_block(nn.Module):
    def __init__(self,inc,filters,outc):
        super(yolo_block,self).__init__()

        self.conv1 = Conv2D_BN_Leaky(inc,filters,1)
        self.conv2 = Conv2D_BN_Leaky(filters,filters*2,3)
        self.conv3 = Conv2D_BN_Leaky(filters*2,filters,1)
        self.conv4 = Conv2D_BN_Leaky(filters,filters*2,3)
        self.conv5 = Conv2D_BN_Leaky(filters*2,filters,1)
        self.conv6 =Conv2D_BN_Leaky(filters,filters*2,3)
        self.conv7 =Conv2D(filters*2,outc,1,1,0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        y = self.conv6(x)
        y = self.conv7(y)
        return x,y


class upsample(nn.Module):
    def __init__(self, inc, outc, ):
        super(upsample, self).__init__()
        self.conv = Conv2D_BN_Leaky(inc, outc, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x,scale_factor=2)
        return x


class yolo_body(nn.Module):

    def __init__(self, num_classes):
        super(yolo_body, self).__init__()
        self.conv = Conv2D_BN_Leaky(3, 32)
        self.resblck1 = resblock(32, 64, 1)
        self.resblck2 = resblock(64, 128, 2)
        self.resblck3 = resblock(128, 256, 8)
        self.resblck4 = resblock(256, 512, 8)
        self.resblck5 = resblock(512, 1024, 4)
        self.yolo_block1 = yolo_block(1024, 512, 3 * (num_classes + 5))
        self.yolo_block2 = yolo_block(768, 256, 3 * (num_classes + 5))
        self.yolo_block3 = yolo_block(384, 128, 3 * (num_classes + 5))
        self.upsample1 = upsample(512, 256)
        self.upsample2 = upsample(256, 128)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblck1(x)
        x = self.resblck2(x)
        out1 = self.resblck3(x)
        out2 = self.resblck4(out1)
        out3 = self.resblck5(out2)

        out, y1 = self.yolo_block1(out3)

        out = self.upsample1(out)
        out = torch.cat((out, out2), 1)
        out, y2 = self.yolo_block2(out)

        out = self.upsample2(out)
        out = torch.cat((out, out1), 1)
        out, y3 = self.yolo_block3(out)
        return [y1, y2, y3]
# 定义模型加载函数
    def load_weight(self,path):
        old = torch.load(path)
        model_dic = self.state_dict()
        for k in model_dic.keys():
            if k in old.keys() and model_dic[k].size() == old[k].size():
                model_dic[k] = old[k]
            else:
                print('layer:',k,'miss match')
        self.load_state_dict(model_dic)

def GHMC_loss(out_put,y_true):
    bins = 10
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6
    weights = torch.zeros_like(out_put)
    g = torch.abs(torch.sigmoid(out_put) - y_true)
    tot = 1
    for k in out_put.size():
        tot *= k
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            weights[inds] = tot / num_in_bin
    weights /= 10
    loss = F.binary_cross_entropy_with_logits(out_put,y_true,weight=weights,reduction='none')
    return loss

def focal_loss(out_put,y_true):
    loss = ((torch.sigmoid(out_put))**2*(1-y_true)+ (1-torch.sigmoid(out_put))**2*y_true)\
            *F.binary_cross_entropy_with_logits(out_put, y_true, reduction='none')
    return loss

def yolo_loss(out_put,y_true,num_classes,anchors,input_shape,CUDA,loss_function = 'None',print_loss = True):
    m = len(out_put)
    loss = 0

    for i in range(m):
        scal = out_put[i].shape[-1]
        X = out_put[i].reshape(-1,3,num_classes+5,scal,scal)
        Y = torch.from_numpy(y_true[i])
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchor = [anchors[anchor_mask[i][n]] for n in range(3)]
        anchor = torch.FloatTensor(anchor)
        if CUDA:
            X = X.cuda()
            Y = Y.cuda()
            anchor = anchor.cuda()
        x = anchor[:, 0].view(-1, 1).repeat(1, scal * scal).reshape(3, scal, scal).unsqueeze(1)
        y = anchor[:, 1].view(-1, 1).repeat(1, scal * scal).reshape(3, scal, scal).unsqueeze(1)
        anchor_xy = torch.cat((x, y), 1)
        w_h = torch.exp(Y[..., 2:4, :, :]) * anchor_xy
        boxes_loss_scal = 2 - w_h[...,0,:,:]*w_h[...,1,:,:]/(input_shape[0]*input_shape[1])
        boxes_loss_scal = torch.unsqueeze(boxes_loss_scal, 2)
        object_mask = torch.unsqueeze(Y[...,4,:,:],2)
        xy_loss = F.binary_cross_entropy_with_logits(X[...,:2,:,:],Y[...,:2,:,:],reduction='none')*object_mask*boxes_loss_scal
        wh_loss = F.mse_loss(X[...,2:4,:,:],Y[...,2:4,:,:],reduction='none')*object_mask*boxes_loss_scal
        if loss_function == 'focal_loss':
            confidence_loss = focal_loss(X[...,4,:,:],Y[...,4,:,:])
            cla_loss = focal_loss(X[..., 5:, :, :], Y[..., 5:, :, :])*object_mask
        elif loss_function == 'GHM_loss':
            confidence_loss = GHMC_loss(X[..., 4, :, :], Y[..., 4, :, :])
            cla_loss = GHMC_loss(X[...,5:,:,:],Y[...,5:,:,:])*object_mask
        else:
            confidence_loss = F.binary_cross_entropy_with_logits(X[...,4,:,:],Y[...,4,:,:],reduction='none')
            cla_loss = F.binary_cross_entropy_with_logits(X[...,5:,:,:],Y[...,5:,:,:],reduction='none')*object_mask
        scal_loss = xy_loss.sum()+wh_loss.sum()+confidence_loss.sum()+cla_loss.sum()
        loss += scal_loss
        if print_loss:
            print('loss:{:<10.3f}'.format(scal_loss),'xy_loss:{:<10.3f}'.format(xy_loss.sum()),
                  'wh_loss:{:<10.3f}'.format(wh_loss.sum()),'confidence_loss:{:<10.3f}'.format(confidence_loss.sum()),
                  'cla_loss:{:<10.3f}'.format(cla_loss.sum()))
    return loss
