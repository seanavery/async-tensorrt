import threading
import time
from queue import Queue
from ssd.Processor2 import Processor
import cv2

# global data
boxes = []

lock = threading.Lock()

def processor(p):
    global boxes
    while True:
        val = q.get()
        if val is not None:
            boxes, confs, clss = p.detect(val)

        with lock:
            boxes = boxes
            print('val', val)
            print('boxes', boxes)
            print('p', p)

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
            cv2.imshow("Camera", frame)
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("could not open camera")

if __name__ == '__main__':
    q = Queue()
    print('q', q)
    p = Processor()
    thread = threading.Thread(target=processor, kwargs={ 'p': p })
    thread.daemon = True
    thread.start()
    # wait three seconds for processor to boot up
    camera_stream()
