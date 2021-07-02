"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from collections import OrderedDict
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from skimage import io
from torch.autograd import Variable

import craft_utils
import file_utils
import imgproc
from craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--result_folder', default="./result/", type=str, help='result save directory')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = args.result_folder
if os.path.isdir(result_folder):
    shutil.rmtree(result_folder)

os.mkdir(result_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def generate_pseudo_label_ui(image_path: str, labels: List[str], label_to_id: dict,  bboxes: np.ndarray, dirname: str) -> None:
    """ save text detection result one by one.

    Args:
        img_file: image file name
        labels: ground truth label
        label_to_id: map label into id
        bboxes: predict bounding boxes
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        dirname: output directory
    Return:
        None
    """

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # result directory
    res_file = dirname + "/" + filename + '.txt'
    img_file = dirname + "/" + filename + '.jpg'


    with open(res_file, 'w') as f:
        for i, box in enumerate(bboxes):
            # print(f"box:{box}")
            # poly = np.array(box).astype(np.int32).reshape((-1))
            # print(f"poly:{poly}")
            x1, y1, x2, y2 = int(box[0][0]), int(box[0][1]), int(box[2][0]), int(box[2][1])
            ui_format = f"{labels[i]} 0 {label_to_id[labels[i]]} -1 1 {x1/width} {y1/height} {x2/width} {y2/height} 0\n"
            f.write(ui_format)

            # poly = poly.reshape(-1, 2)
            # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    #         img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Save result image
    cv2.imwrite(img_file, img)


def critera(bboxes: np.ndarray, labels: str) -> Tuple[bool, np.ndarray]:
    """Check predict boxes is good enough for pseudo labeling

    Filter small boxes and compare #boxes == #labels

    Args:
        boxes: predict boxes.
        labels: ground truth labels.
    
    Return:
        bool: Can be pseudo label or not
        np.ndarray: predict and filtered boxes
    """
    if type(bboxes) != np.ndarray or bboxes.size == 0:
        return False , bboxes

    dim = bboxes.shape[0]
    bboxes = bboxes.reshape(dim, -1)
    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    bboxes = bboxes[(area >= 0.75*area.mean()) & (area <= 1.25*area.mean())]

    if len(labels) == len(bboxes):
        return True, bboxes
    else:
        return False, bboxes


if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))

    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    label_to_id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        # # save score text
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)

        labels = list(filename.split("_")[0])
        flag, bboxes = critera(bboxes, labels)
        if flag == True:
            generate_pseudo_label_ui(image_path, labels, label_to_id, polys, result_folder)

    print("elapsed time : {}s".format(time.time() - t))
