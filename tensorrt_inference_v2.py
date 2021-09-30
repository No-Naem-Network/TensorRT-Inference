import threading
import cv2
import numpy as np
import argparse
import sys
import os
import time
import torch
import torchvision
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

import tensorrt as trt
print(trt.__version__)
trt.init_libnvinfer_plugins(None, '')
import pycuda.autoinit
import pycuda.driver as cuda

INPUT_H = 640  #defined in decode.h
INPUT_W = 640
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4

# # ---------- TRT ---------- #

ONNX_FILE_PATH='models/weights/Retinaface_m25_dynamic_batch.onnx'

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = - trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            print(host_mem)
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

class FaceDetectionRetinaTRT():
    """[RetinaFace Detector with TRT and fixed input size]

    Args:
        FaceDetectionInterface ([Interface]): [Interface]
    """    
    def __init__(self, engine_file_path):
        print("Start loading RetinaFace Detector")
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        self.EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        
        self.output_shapes = [(1, 16800, 4), (1, 16800, 10), (1, 16800, 2)]
        
        engine = self.get_engine(engine_file_path)
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        self.engine = engine
        self.context = context
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        print("Finish loading RetinaFace Detector")
        print("-"*40)

    def preprocess_image(self, input_image_path):
        """
        description: Read an image from image path, resize and pad it to target size,
                     normalize to [0,1],transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = cv2.imread(input_image_path)
        h, w, c = image_raw.shape
        image_rsz = cv2.resize(image_raw, (INPUT_W, INPUT_H))

        image = image_rsz.astype(np.float32)

        # HWC to CHW format:
        image -= (104, 117, 123)
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_rsz, h, w

    def find_bbox(self, image):
        """[Find ]

        Args:
            image ([type]): [description]
        """
        self.cfx.push()   
        boxes, lanndms = None, None
        img_process, image_rsz, h, w = self.preprocess_image(image)

        engine = self.engine
        context = self.context
        inputs = self.inputs
        outputs = self.outputs
        bindings = self.bindings
        stream = self.stream

        context.set_binding_shape(0, (1, 3, INPUT_H, INPUT_W))

        # inputs[0].host = img_process
        np.copyto(inputs[0].host, img_process.ravel())
        trt_outputs = self.do_inference_v2(context, bindings, inputs, outputs, stream)
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        
        print(trt_outputs)
        # boxes, lanndms = self.retina_postproc.process(image, trt_outputs)
        self.cfx.pop()
        return boxes, lanndms

    def align_face(self, image, bbox, landmark):
        """[summary]

        Args:
            image ([type]): [description]
            bbox ([type]): [description]
            landmark ([type]): [description]

        Returns:
            [type]: [description]
        """        
        dets = np.concatenate((bbox, landmark))
        aligned_face = self.retina_postproc._process_align(image, dets)

        return aligned_face

    def get_engine(self, engine_file_path):
        """[Loading TRT]
        """

        def build_engine():
            """[Build engine from Onnx for inference]
            """

            with trt.Builder(self.TRT_LOGGER) as builder:
                config_builder = builder.create_builder_config()
                profile = builder.create_optimization_profile()
                profile.set_shape("input0", (1, 3, 640, 640), (4, 3, 640, 640), (8, 3, 640, 640))
                config.add_optimization_profile(profile)
                network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                with trt.OnnxParser(network, self.TRT_LOGGER) as parser:
                    builder.max_workspace_size = 1 << 28 # 256MB
                    builder.max_batch_size = 1
                    # builder.fp16_mode = True
                    # builder.strict_type_constraints = True
                    
                    # Parse model file

                    if not os.path.exists(ONNX_FILE_PATH):
                        print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(ONNX_FILE_PATH))
                        exit(0)

                    print('Loading ONNX file from path {}...'.format(ONNX_FILE_PATH))

                    with open(ONNX_FILE_PATH, 'rb') as model:
                        print('Beginning ONNX file parsing')
                        if not parser.parse(model.read()):
                            print ('ERROR: Failed to parse the ONNX file.')
                            for error in range(parser.num_errors):
                                print (parser.get_error(error))
                            return None

                    # network.get_input(0).shape = [1, 3, 640, 640]
                    # network.get_input(0).shape = [1, 3, 1920, 1080]
                    network.get_input(0).shape = [-1, 3, 640, 640]
                    print('Completed parsing of ONNX file')
                    print('Building an engine from file {}; this may take a while...'.format(ONNX_FILE_PATH))

                    engine = builder.build_cuda_engine(network)
                    print("Completed creating Engine")
                    with open(engine_file_path, 'wb') as f:
                        f.write(engine.serialize())
                    return engine


        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, 'rb') as f , trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        """[This function is generalized for multiple inputs/outputs for full dimension networks.
            Inputs and outputs are expected to be lists of HostDeviceMem objects.]

        Returns:
            [list]: [Detection output]
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        # context.set_optimization_profile_async(0, stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

class myThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)


if __name__ == "__main__":
    # load custom plugins,make sure it has been generated
    # PLUGIN_LIBRARY = "build/libdecodeplugin.so"
    # ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = "models/weights/retinaface.trt"

    retinaface = FaceDetectionRetinaTRT(engine_file_path)
    input_image_paths = ["curve/test.jpg"]
    # for i in range(10):
    for input_image_path in input_image_paths:
        # create a new thread to do inference
        thread = myThread(retinaface.find_bbox, [input_image_path])
        thread.start()
        thread.join()

    # destroy the instance
    retinaface.destroy()