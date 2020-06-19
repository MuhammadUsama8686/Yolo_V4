from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import argparse

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--image', required=True,
                  help='image path')
parser.add_argument('-o', '--output',required=True,
                  help='output_path')
args = parser.parse_args()


def convertBack(x, y, w, h,md_w,md_h,orig_w,orig_h):
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)

    xmin = int((xmin*orig_w)/md_w)
    xmax = int((xmax*orig_w)/md_w)
    ymin = int((ymin*orig_h)/md_h)
    ymax = int((ymax*orig_h)/md_h)
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img,md_w,md_h):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h),md_w,md_h,img.shape[1],img.shape[0])
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 3)
        name = detection[0].decode()
        pred = float(detection[1])
        cv2.putText(img,
                    name +
                    " [" + str(round(pred * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 0], 2,cv2.LINE_AA)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():

    global metaMain, netMain, altNames
    configPath = "yolov4-obj.cfg"
    weightPath = "backup/yolov4-obj_2000.weights"
    metaPath = "coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    # while True:
    frame_read = cv2.imread(args.image)    
    prev_time = time.time()

    frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

    detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
    image = cvDrawBoxes(detections, frame_read,darknet.network_width(netMain),darknet.network_height(netMain))

    print(1/(time.time()-prev_time))
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    cv2.imwrite(args.output,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    YOLO()
