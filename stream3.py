import threading
import time
from queue import Queue
from ssd.Processor3 import Processor
import cv2
import pycuda.driver as cuda

# global data
boxes = []

lock = threading.Lock()

def processor():
    cuda_ctx = cuda.Device(0).make_context()
    p = Processor()
    global boxes
    while True:
        val = q.get()
        if val is not None:
            bxs, confs, clss = p.detect(val)

        with lock:
            print('val', val)
            print('boxes', bxs)
            print('p', p)
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
            cv2.imshow("Camera", frame)
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("could not open camera")

if __name__ == '__main__':
    cuda.init()
    q = Queue()
    print('q', q)
    thread = threading.Thread(target=processor)
    thread.daemon = True
    thread.start()
    # wait three seconds for processor to boot up
    time.sleep(10)
    camera_stream()
