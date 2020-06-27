import sys
import time
import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

class Processor():  
    """
        Processes numpy frame
    """
    def __init__(self):
        # setup tensorrt engine
        trt_logger = trt.Logger(trt.Logger.INFO)
        TRTbin = 'models/ssd-mobilenet-v2-coco.trt'

        # load plugins
        trt.init_libnvinfer_plugins(trt_logger, '')

        # load engine
        with open(TRTbin, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # create context
        self.host_inputs = []
        self.host_outputs = []
        self.cuda_inputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                    self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
            
            self.context = self.engine.create_execution_context()

    def __del__(self):
        """ free cuda memory """
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs

    def detect(self, frame):
        pre_start = time.time()
        # 1. pre-process image
        resized = self.pre_process(frame)
        print('resized', resized)
        # flatten np image
        np.copyto(self.host_inputs[0], resized.ravel()) 

        # copy buffer into cuda, serialize via stream
        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        pre_end = time.time()

        # execute inference async
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings, # input/output buffer addresses
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)

        self.stream.synchronize()
        output = self.host_outputs[0]

        # post process output
        return self.post_process(frame, output, confidence_threshold=0.3)

    def pre_process(self, frame):
        # convert to 300 * 300
        # TODO: check if hardware accelerated
        frame = cv2.resize(frame, (300, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1)).astype(np.float32)
        frame *= (2.0/255.0)
        frame -= 1.0

        return frame
   
    def infer(self, frame):
        return True

    def post_process(self, frame, output, confidence_threshold):
        img_h, img_w, _ = frame.shape
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), 7):
            confidence = float(output[prefix+2])
            if confidence < confidence_threshold:
                continue
            x1 = int(output[prefix+3] * img_w)
            y1 = int(output[prefix+4] * img_h)
            x2 = int(output[prefix+5] * img_w)
            y2 = int(output[prefix+6] * img_h)
            cls = int(output[prefix+1])
            boxes.append((x1, y1, x2,  y2))
            confs.append(confidence)
            clss.append(cls)
        return boxes, confs, clss

