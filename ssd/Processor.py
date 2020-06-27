import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2 
import sys

class SSD():
    def __init__(self):
        # initialize logger for debugging
        trt_logger = trt.Logger(trt.Logger.INFO)
        
        # load libnvinfer plugins
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
        trt.init_libnvinfer_plugins(trt_logger, '')
        
        # instantiate TensorRT engine
        trt_model = 'models/ssd-mobilenet-v2-coco.trt'
        with open(trt_model, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # create context
        self.stream = cuda.Stream()

        # memory allocations for input/output layers
        # binding Input
        # binding NMS
        # binding NMS_1
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * \
                    engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)   
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({ 'host': host_mem, 'cuda': cuda_mem })
            else:
                self.outputs.append({ 'host': host_mem, 'cuda': cuda_mem })

        self.context = engine.create_execution_context()

    def detect(self, frame):
        resized = self.preprocess(frame)
        print('resized', resized)
        output = self.infer(resized)
        print('output', output)
        print('output shape', output.shape)
        sys.exit()
        boxes, confs, clss = self.postprocess(frame, output)
        print('boxes', boxes)
        print('confs', confs)
        print('clss', clss)
    
    def preprocess(self, frame):
        frame = cv2.resize(frame, (300, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1)).astype(np.float32)
        frame *= (2.0/255.0)
        frame -= 1.0
        return frame

    def infer(self, frame):
        # flatten input image
        np.copyto(self.inputs[0]['host'], frame.ravel())
        
        # execute inference 
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        
       
        # fetch outputs from gpu
        cuda.memcpy_dtoh_async(
            self.outputs[1]['host'], self.outputs[1]['cuda'], self.stream)
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'], self.outputs[0]['cuda'], self.stream)
     
        # wait for kernel completion before host access
        self.stream.synchronize()
 
        return self.outputs[0]['host']

    def postprocess(self, frame, output):
        print('frame.shape', frame.shape)
        h, w, _ = frame.shape
        boxes, confs, clss = [], [], []
        for prefix in range(0, len(output), 7):
            confidence = float(output[prefix+2])
            print('confidence', confidence)
            if confidence > 0.0:
                print('found one!!!', confidence)
                sys.exit()
            if confidence < 0.1:
                continue
            x1 = int(output[prefix+3] * w)
            y1 = int(output[prefix+4] * h)
            x2 = int(output[prefix+5] * w)
            y2 = int(output[prefix+6] * h)
            cls = int(output[prefix+1])
            boxes.append((x1, y1, x2, y2))
            confs.append(confidence)
            clss.append(cls)

        return boxes, confs, clss

