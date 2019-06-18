from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


"""
get feature map to 2-D tensor
and normalize each value

Parameters
----------
prediction : pytorch tensor
    output feature map
inp_dim : int
    input image
anchors : int
num_classes : int

bbox_attrs: [double, double, double, double, double]
    x, y, w, h, confidence score
"""
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # anchor size fit to actual image size, so divide by stride to fit feature map size
    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    # tensor shape to (batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size) 
    prediction = prediction.transpose(1, 2).contiguous() # transpose dim 1 and dim 2, and on memory 
    # tensor shape to (batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # add the centre_X, centre_Y. and object confidence score
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the center offsets
    """
    a: [[0,1,2,...], [0,1,2,...]]
    b: [[0,0,0,...], [1,1,1,...]]
    """
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    # convert float type 1 dim tensor
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # concanticate(x, y) and repeat num_anchors times, reshape dim 2 and extend dim
    """
    x_y_offset:
        [
         [[0., 0.],
          [0., 0.],
          [0., 0.],
          [1., 0.],
          [1., 0.],
          [1., 0.],
          [2., 0.],
          [2., 0.],
          [2., 0.],
          [0., 1.],
         ...
    """
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueese(0)

    prediction[:,:,:2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    # apply sigmoid activation to the class scores
    # 5: (tx, ty, tw, th, objectness score)
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    # resize feature map size to input image size
    prediction[:,:,:4] *= stride

    return prediction






