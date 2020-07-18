import sys
import tensorrt as trt

if __name__ == '__main__':
    model = '../models/ResNet101-DUC-7.onnx'
    logger = trt.Logger(trt.Logger.VERBOSE) 
    explicit_batch = [ 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) ]
    with trt.Builder(logger) as builder, builder.create_network(*explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        with open(model, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit()

        engine = builder.build_cuda_engine(network)

        with open('ResNet101-DUC-7.trt', 'wb') as f:
            f.write(engine.serialize())


