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
import random
import pickle as pkl
import argparse

def arg_parse():
    """
    Parse arguments to the detect module 
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection')
    parser.add_argument("--video", dest='video',
            help="Video to run detection upon", default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset",
            help="Dataset on which the network has been trained", default="pascal")
    parser.add_argument("--confidence", dest="confidence",
            help="Object Confidence to filter predictions", default=0.5, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
            help="NMS Threshhold", default=0.4, type=float)
    parser.add_argument("--cfg", dest="cfgfile",
            help="Config file", default="cfg/yolov3.cfg", type=str)
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
    num_classes = 80
    bbox_attrs = 5 + num_classes

    print(f"Loading network.....")

if __name__ == '__main__':
    main()


