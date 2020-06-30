import cv2
import random
import colorsys
from utils.coco_classes import COCO_CLASSES_LIST
import sys

class Visualizer():
    def __init__(self):
        self.color_list = self.gen_colors()

    def gen_colors(self):
        # generate random hues
        hsvs = []
        for x in range(len(COCO_CLASSES_LIST)):
            hsvs.append([float(x) / len(COCO_CLASSES_LIST), 1., 0.7])
        random.seed(3344)
        random.shuffle(hsvs)
        
        # convert hsv to rgb values
        rgbs = []
        for hsv in hsvs:
            (h, s, v) = hsv
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append(rgb)

        # convert to bgr and (0-255) range
        bgrs = []
        for rgb in rgbs:
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            bgrs.append(bgr)

        return bgrs

    def draw(self, frame, boxes, confs, clss):
        overlay = frame.copy()
        for bb, cf, cl in zip(boxes, confs, clss):
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            cls = COCO_CLASSES_LIST[cl]
            color = self.color_list[cl]
            print('cls', cls) 
            if cls == 'dining table' or cls == 'suitcase':
                print("here")
                continue
            print('color', color)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
            cv2.putText(frame, cls, (x_min + 20, y_min + 20), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
        
        alpha = 0.4

        return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

