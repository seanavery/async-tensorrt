import time
import cv2

def stream_camera():
    pipeline = (
        "nvarguscamerasrc wbmode=2 !"
        "nvvidconv flip-method=2 ! "
        "videoconvert ! video/x-raw, format=(string)BGR !"
        "appsink"
    )
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    print('video_capture', video_capture)
    if video_capture.isOpened():
        prev_time = 0
        while True:
            window = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            return_value, frame = video_capture.read()
            post_time = time.time()
            print("time between frames:", (post_time - prev_time) * 1000)
            prev_time = post_time
            show_time_start = time.time()
            cv2.imshow("Camera", frame)
            show_time_end = time.time()
            print('imshow time:', (show_time_end-show_time_start) * 1000)
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("could not open camera")

if __name__ == "__main__":
    stream_camera()
