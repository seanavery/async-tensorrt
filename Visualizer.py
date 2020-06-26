import cv2
import random
import colorsys

COCO_CLASSES_LIST = [
    'background',  # was 'unlabeled'
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

class Visualizer():
    def __init__(self, color):
        # TODO: green gradient
        self.color = color
        self.color_map = self.gen_colors()

    def gen_colors(self):
        # generate random hues
        hsvs = []
        for x in range(len(COCO_CLASSES_LIST)):
            hsvs.append([float(x) / len(COCO_CLASSES_LIST), 1., 1.])
        random.seed(13414)
        random.shuffle(hsvs)

    def draw(self, frame, boxes, confs, clss):
        overlay = frame.copy()
        for bb, cf, cl in zip(boxes, confs, clss):
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            print('cl', cl, COCO_CLASSES_LIST[cl])
            print('cf', cf)
            print('x_min', x_min)
            print('y_min', y_min)
            print('x_max', x_max)
            print('y_max', y_max)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), self.color, -1)
        
        alpha = 0.4

        return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

