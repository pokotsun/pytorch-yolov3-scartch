from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet

import pandas as pd
import colorsys
import random
import pickle as pkl
import argparse

def arg_parse():
    """
    Parse arguments to the detect module 
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection')
    parser.add_argument("--dataset", dest="dataset",
            help="Dataset on which the network has been trained", default="pascal")
    parser.add_argument("--confidence", dest="confidence",
            help="Object Confidence to filter predictions", default=0.5, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
            help="NMS Threshhold", default=0.4, type=float)
    parser.add_argument("--cfg", dest="cfgfile",
            help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest="weightsfile", 
            help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest="reso",
            help="Input resolution of the network. Increase to increase accuracy. Decrease to increase learning speed",
            default=416, type=int)
    return parser.parse_args()

def main():
    args = arg_parse()
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    start = 0
    CUDA = torch.cuda.is_available()
    
    classes = load_classes("data/coco.names")
    num_classes = len(classes)

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # if there's a GPU available, put the model on GPU
    if CUDA:
        model.cuda()

    # set the model in evaluation mode
    model.eval()

    def write(x, img, color):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 4)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), thickness=1)

    # detection phaase
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Cannot capture source"

    frames = 0
    start = time.time()
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 200), int(x[1] * 200), int(x[2] * 200)), colors))
    np.random.seed(10000)
    np.random.shuffle(colors)
    np.random.seed(None) # reset seed to default.
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, dsize=(1280, 960))
            img = prep_image(frame, inp_dim)
            print(f"IMG_SHAPE: {img.shape}")
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                outputs = model(Variable(img, volatile=True), CUDA)
            outputs = write_results(outputs, confidence, num_classes, nms_conf=nms_thresh)

            if outputs != None:
                im_dim = im_dim.repeat(outputs.size(0), 1)
                scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

                outputs[:, [1,3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1,1)) / 2
                outputs[:, [2,4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1,1)) / 2

                outputs[:, 1:5] /= scaling_factor

                for i in range(outputs.shape[0]):
                    outputs[i, [1,3]] = torch.clamp(outputs[i, [1,3]], 0.0, im_dim[i, 0])
                    outputs[i, [2,4]] = torch.clamp(outputs[i, [2,4]], 0.0, im_dim[i, 1])

                for output in outputs:
                    color = colors[int(output[-1])]
                    write(output, frame, color)

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start) ))
        else:
            break

if __name__ == '__main__':
    main()


