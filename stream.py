import time
import cv2

# "nvarguscamerasrc auto-exposure=1 wbmode=1 saturation=1.5 tnr-mode=2 tnr-strength=2 tnr-mode=     2 ee-mode=2 ee-strength=0.5 !"

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    flip_method=2,
):
    return (
        "nvarguscamerasrc wbmode=2 !"
        # "video/x-raw(memory:NVMM), "
        # "width=(int)%d, height=(int)%d, "
        # "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        # "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            flip_method,
        )
    )


def stream_camera():
    pipeline = gstreamer_pipeline()
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
    # initiate gstreamer pipeline to CSI camera
    stream_camera()
