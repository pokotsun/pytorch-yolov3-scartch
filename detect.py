from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguments to the detect module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', 
            help="Image / Directory containing images to perform detection upon",
            default = "imgs", type=str)
    parser.add_argument("--det", dest='det', 
            help="Image / Directory to store detections to",
            default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest="cfgfile", help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest="weightsfile", help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest="reso", help="Input resolution of the network. Increase to increase accuracy. Descrease to increase speed", default="416", type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    classes = load_classes("data/coco.names")
    num_classes = len(classes)

    # setup the neural network
    print("Loading network...")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded!!")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    # if there's a GPU available, put the model on GPU
    if CUDA:
        model.cuda()

    # set the model in evaluation(prediction) mode
    model.eval()

    read_dir = time.time() # check point time
    # detection phase
    try:
        imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(os.path.join(os.path.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()
    
    # if detection result path is not found, create it
    if not os.path.exists(args.det):
       os.makedirs(args.det)

    load_batch = time.time() # check point time

    loaded_ims = [cv2.imread(x) for x in imlist]

    # pytorch variables for images
    im_batches = list(map(prep_image, loaded_ims, [inp_dim for _ in range(len(imlist))] ))

    # list containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims] 
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size : min( (i+1) * batch_size, len(im_batches) )]))
                for i in range(num_batches)]

   
    write = False
    start_det_loop = time.time()

    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        # apply offsets to the result predictions
        # transform the predictions as described in the YOLO paper
        # flatten the prediction vector
        # B x (bbox cord x No. of anchors) x grid_w x grid_h -> B x bbox x (all the boxes)
        # put every proposed box as a row

        # if volatile = True, not store compute graph
        prediction = model(Variable(batch, volatile=True), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(imlist[i*batch_size: min((i+1) * batch_size), len(imlist)]):
                im_id = i * batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("---------------------------------------------------------------")
            continue
        
        prediction[:, 0] += i * batch_size # transform the attribute from index in batch to index in imlist

        if not write:
            output = prediction
            write = True
        else:
            output = torch.cat((output, prediction))

        print(output)
        for im_num, image in enumerate(imlist[i*batch_size: min((i+1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0] == im_id)] 
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("---------------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()

    # draw BBox to images
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1, 1)

    output[:, [1,3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2,4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()

    class_load = time.time()
    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()

    def write(x, results, color):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}.format(classes[cls])"
        cv2.rectangle(img, c1, c2, color, 1) # draw rectangle
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 225, 225], 1)
        return img
    
    list(map(lambda x: write(x, loaded_ims), output))

    det_names = pd.Seriese(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, loaded_ims))

    end = time.time()

    print("SUMMARY")
    print("--------------------------------------------")
    print("{:25s}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))

    torch.cuda.empty_cache()



