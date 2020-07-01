import threading
import time
from queue import Queue
from ssd.Processor import Processor
from ssd.Visualizer import Visualizer
import cv2
import pycuda.driver as cuda

# global data
boxes = []
confs = []
clss = []

lock = threading.Lock()

def processor():
    cuda_ctx = cuda.Device(0).make_context()
    p = Processor()
    global boxes
    global confs
    global clss

    while True:
        val = q.get()
        if val is not None:
            bxs, cfs, cls = p.detect(val)

        with lock:
            boxes = bxs
            confs = cfs
            clss = cls
    del p
    del cuda_ctx

def camera_stream():
    pipeline = (
        "nvarguscamerasrc wbmode=1 !"
        "nvvidconv flip-method=2 ! "
        "videoconvert ! video/x-raw, format=(string)BGR !"
        "appsink"
    )
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        modulus = 3
        counter = 0
        while True:
            window = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            _, frame = video_capture.read()
            counter = counter + 1
            if counter % modulus == 0:
                q.put(frame)
            frame = vis.draw(frame, boxes, confs, clss)
            cv2.imshow("Camera", frame)
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("could not open camera")

if __name__ == '__main__':
    vis = Visualizer()
    cuda.init()
    q = Queue()
    thread = threading.Thread(target=processor)
    thread.daemon = True
    thread.start()
    # wait three seconds for processor to boot up
    time.sleep(10)
    camera_stream()
