from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


"""
write results that applied Non Maximum Suppression

Parameters
----------
prediction : pytorch tensor
    output feature map[batch_size, grid_size x grid_size x num_anchors, box_attrs]
confidence : float
    objectness score threshold
num_classes : int
nms_conf: float
    NUMS IoU threshold

box_attrs: [double, double, double, double, double]
    x, y, w, h, confidence score

Return
------
(all true detections) x 
(index of the image in batch, 4 corner cordinates, 
 objectness-score, score of class with maximum confidence, index of class)
"""
def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    height_half = prediction[:, :, 2]/2
    width_half = prediction[:, :, 3]/2
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = prediction[:,:,0] - height_half # top-left corner x
    box_corner[:,:,1] = prediction[:,:,1] - width_half # top-left corner y 
    box_corner[:,:,2] = prediction[:,:,0] + height_half # bottom-right corner x
    box_corner[:,:,3] = prediction[:,:,1] + width_half # bottom-right corner y 

    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    output = prediction.new(1, prediction.size(2) + 1)
    # if write=False, output have not been initialized
    write = False

    for idx in range(batch_size):
        # select the image from the batch
        image_pred = prediction[idx]

        # get the class having maximum score, and the index of that class
        # get rid of num_classes softmax scores
        # add the class index and the class score of class having maximum score
        # return (scores, indices)
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        """
        image_pred: pytorch tensor
            [   grid_size x grid_size x num_anchors, 
                [top-left x, top-left y, bottom-right x, bottom-right y, objectness_score], 
                confidence_score, confidence_class_index
            ]
        """

        # get rid of the zero entries
        non_zero_indices = (torch.nonzero(image_pred[:,4]))
        # if objectness is zero, continue next batch(image)
        try:
            image_pred_ = image_pred[non_zero_indices.squeeze(), :]
        except:
            continue

        # get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1].long()).half() # -1 holds the class index

        # apply NMS each classes 
        for cls in img_classes:
            # get the detections with current class
            cls_mask = image_pred_*(image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(cls_mask[:, -2]).squeeze() # get indices of nonzero confidence score
            
            image_pred_cls = image_pred_[class_mask_idx].view(-1, 7) # get nonzero anchors and reshape (n, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            image_pred_cls_idx_sorted = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_cls_sorted = image_pred_cls[image_pred_cls_idx_sorted]
            num_image_pred_classes = image_pred_cls_sorted.size(0) # num of each grid anchor box size and something was appeared 

            # if NMS has to be done
            if nms:
                # for each detection result
                for i in range(num_image_pred_classes):
                    # get the IoUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_cls_sorted[i].unsqueeze(0), image_pred_cls_sorted[i+1:])

                    except ValueError:
                        break

                    except IndexError:
                        break

                    # zero out all the detections that have IoU > threshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_cls_sorted[i+1:] *= iou_mask

                    # remove the non-zero entries
                    non_zero_indices = torch.nonzero(image_pred_cls_sorted[:, 4]).squeeze()
                    image_pred_cls_sorted = image_pred_cls_sorted[non_zero_indices].view(-1, 7)

            # concatenate the batch_id of the image to the detection result.
            # this helps up identify which image does the detection corresponds to
            # we use a linear stracture to hold All the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra column
            batch_idx = image_pred_cls_sorted.new(image_pred_cls_sorted.size(0), 1).fill_(idx)
            seq = batch_idx, image_pred_cls_sorted

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    return output

                
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

"""
returns the IoU of two bounding boxes
"""
def bbox_iou(box1, box2):
    # get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    #NOTE why +1?
    # intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1+1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b1_x2 + 1) * (b2_y2 - b2_y1 + 1)

    union_area = b1_area + b2_area - inter_area

    iou = inter_area / union_area

    return iou

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
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # anchor size fit to actual image size, so divide by stride to fit feature map size
    anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

    # tensor shape to (batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size) 
    prediction = prediction.transpose(1, 2).contiguous() # transpose dim 1 and dim 2, on memory 
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
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    # apply sigmoid activation to the class scores
    # 5: (tx, ty, tw, th, objectness score)
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    # resize feature map size to input image size
    prediction[:,:,:4] *= stride

    return prediction






