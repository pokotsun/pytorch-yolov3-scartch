from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# resize and padding for not changing aspect ratio
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    # when enlarge image, INTER_CUBIC is best(a little slow)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((h, w, 3), 128)
    canvas[(h - new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network
    OpenCV BGR numpy tensor to Tensorflow RGB Variable 
    """
    #img = cv2.resize(img, (inp_dim, inp_dim), interpolation=cv2.INTER_CUBIC)
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy() # (height, width, channel) to (channel, height, weight)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) # convert to torch Tensor and normalize
    
    return img

def load_classes(names_file):
    names = []
    with open(names_file, 'r') as f:
        names = f.read().split("\n")[:-1]
    return names

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    # NOTE why copy tensor_res and not just return unique_tensor?
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

    # by adding 1, avoid zero division error
    # intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1+1, min=0)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b1_x2 + 1) * (b2_y2 - b2_y1 + 1)

    union_area = b1_area + b2_area - inter_area

    iou = inter_area / union_area

    return iou

"""
write results that applied Non Maximum Suppression

Parameters
----------
prediction : pytorch tensor
    output feature map[batch_size, grid_size * grid_size * num_anchors, box_attrs]
confidence : float
    objectness score threshold
num_classes : int
nms_conf: float
    NUMS IoU threshold

box_attrs: [double, double, double, double, double]
    x, y, w, h, confidence score

Return
------
(all true detections) * 
(index of the image in batch, 4 corner cordinates, 
 objectness-score, score of detected class, index of detected class)
"""
def write_results(prediction, confidence, num_classes, nms=True, nms_conf = 0.4):
    # only objectness > confidence bboxes are evaluated
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

    outputs = None
    # if write=False, output have not been initialized
    write = False

    for img_idx in range(batch_size):
        # select the image from the batch
        image_pred = prediction[img_idx]

        # get the class having maximum score, and the index of that class
        # get rid of num_classes softmax scores
        # add the class index and the class score of class having maximum score
        # return (image_pred, indices, scores)
        max_conf_indices, max_conf_scores = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf_indices = max_conf_indices.float().unsqueeze(1)
        max_conf_scores = max_conf_scores.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf_indices, max_conf_scores)
        image_pred = torch.cat(seq, 1)

        """
        image_pred: pytorch tensor
            [   (grid_size * grid_size * num_anchors) * 
                [top-left x, top-left y, bottom-right x, bottom-right y, objectness_score, 
                most_confidence_class_score, most_confidence_class_idx]
            ]
        """

        # get rid of the zero entries
        non_zero_indices = (torch.nonzero(image_pred[:,4]))

        # if objectness is zero, removed from image_pred
        image_pred = image_pred[non_zero_indices.squeeze(), :]

        # get the various classes detected in the image
        try:
            img_class_indices = unique(image_pred[:,-1]) # -1 holds the class index
        except:
            continue


        # apply NMS each classes 
        for img_class_idx in img_class_indices:
            # get the detections with current class
            # if not current class, it will be 0
            class_mask = image_pred * (image_pred[:, -1] == img_class_idx).float().unsqueeze(1)
            # it will be not -2 ok
            class_mask_indices = torch.nonzero(class_mask[:, -2]).squeeze() # get indices of nonzero confidence score
            
            image_pred_class = image_pred[class_mask_indices].view(-1, 7) # get nonzero anchors and reshape (n, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_indices = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_indices]
            num_image_pred_classes = image_pred_class.size(0) # num of all bounding boxes which something were appeared 

            # if NMS has to be done
            if nms:
                # for each detection result
                for i in range(num_image_pred_classes):
                    # get the IoUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])

                    except ValueError:
                        break

                    except IndexError:
                        break

                    # zero out all the detections that have IoU > threshold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask

                    # remove the non-zero entries
                    # they are very close to i th bbox on image place
                    # and around there, i th bbox is the highest objectness score, so remove others
                    non_zero_indices = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_indices].view(-1, 7)

            # concatenate the batch_id of the image to the detection result.
            # this helps us identify which image does the detection corresponds to
            # we use a linear stracture to hold All the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra column

            batch_indices =image_pred_class.new(image_pred_class.size(0), 1).fill_(img_idx)
            #batch_indices =image_pred_class.new_fill((image_pred_class.size(0), 1), img_idx)
            seq = batch_indices, image_pred_class

            if not write:
                outputs = torch.cat(seq, 1)
                write = True
            else:
                output = torch.cat(seq, 1)
                outputs = torch.cat((outputs, output))

    return outputs

                

"""
get feature map to 2-D tensor
and normalize each value

Parameters
----------
prediction : pytorch tensor
    output feature map
inp_dim : int
    input image (w, h)
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






