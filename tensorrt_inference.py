"""
Use TensorRT's Python api to make inferences.
"""
# -*- coding: utf-8 -*
# import ctypes
import os
import random
import sys
import threading
import time
import argparse
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
INPUT_H = 640  #defined in decode.h
INPUT_W = 640
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='models/weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

def plot_one_box(x, landmark,img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
    param:
        x:     a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness

    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    cv2.circle(img, (int(landmark[0]), int(landmark[1])), 1, (0, 0, 255), 4)
    cv2.circle(img, (int(landmark[2]), int(landmark[3])), 1, (0, 255, 255), 4)
    cv2.circle(img, (int(landmark[4]), int(landmark[5])), 1, (255, 0, 255), 4)
    cv2.circle(img, (int(landmark[6]), int(landmark[7])), 1, (0, 255, 0), 4)
    cv2.circle(img, (int(landmark[8]), int(landmark[9])), 1, (255, 0, 0), 4)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class Retinaface_trt(object):
    """
    description: A Retineface class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(1).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        self.output_shapes = [(1, 16800, 4), (1, 16800, 10), (1, 16800, 2)]
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.device = torch.device("cuda:1")
        start_time = time.time()
        print(engine.num_bindings)
        print(engine.num_optimization_profiles)
        profile_index = context.active_optimization_profile
        print(profile_index)

        for binding in engine:
            print(binding)
            print(engine.get_binding_shape(binding))
            shape = [engine.max_batch_size] + list(engine.get_binding_shape(binding))
            print(shape)
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if size < 0:
                size *= -1
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                for i in range(4):
                    host_inputs.append(host_mem)
                    cuda_inputs.append(cuda_mem)
            else:
                for i in range(4):
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

        print("Time create", time.time() - start_time)

    def infer(self, input_image_path):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        start = time.time()
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        batch = 4
        # Do image preprocess
        context.set_binding_shape(0, (batch, 3, INPUT_H, INPUT_W))
        self.output_shapes = [(batch, 16800, 4), (batch, 16800, 10), (batch, 16800, 2)]
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(
            input_image_path
        )
        input_image_path_2 = "curve/test.jpg"
        input_image_2, image_raw_2, origin_h, origin_w = self.preprocess_image(
            input_image_path_2
        )

        # Copy input image to host buffer
        # input_image = input_image.ravel()
        # np.copyto(host_inputs[0], np.vstack((input_image, input_image, input_image, input_image)).ravel())
        np.copyto(host_inputs[0], input_image.ravel())
        np.copyto(host_inputs[1], input_image_2.ravel())
        # np.copyto(host_inputs[2], input_image.ravel())
        # np.copyto(host_inputs[3], input_image.ravel())
        # Transfer input data  to the GPU.\
        # [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        [cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream) for i in range(4)]
        a = time.time()
        print("Prep time: ", a - start)
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream) for i in range(len(host_outputs))]
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        # Here we use the first row of output in that batch_size = 1
        self.cfx.pop()
        b = time.time()
        print("Time process: ", b - a)
        print(len(host_outputs))
        # trt_outputs = [host_outputs[:batch], host_outputs[batch:batch + batch], host_outputs[batch + batch:]]
        for i in range(batch):
            if  i == 0:
                bbox_outputs = host_outputs[i]
                lmd_outputs = host_outputs[batch+i]
                score_outputs = host_outputs[2*batch+i]
            else:
                bbox_outputs = np.vstack((bbox_outputs, host_outputs[i]))
                lmd_outputs = np.vstack((lmd_outputs, host_outputs[batch+i]))
                score_outputs = np.vstack((score_outputs, host_outputs[2*batch+i]))
        trt_outputs = [bbox_outputs, lmd_outputs, score_outputs]
        print(trt_outputs[0].shape, trt_outputs[1].shape, trt_outputs[2].shape)
        # [host_output for host_output in host_outputs]
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        # Do postprocess
        result_boxes, result_scores, result_landmark = self.post_process(
            trt_outputs, origin_h, origin_w
        )
        end = time.time()
        print("Post process time: ", end - b)
        print("total: ", end - start)

        # Draw rectangles and labels on the original image

        # Save image
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            landmark = result_landmark[i]

            plot_one_box(
                box,
                landmark,
                image_raw,
                label="{}:{:.2f}".format( 'Face', result_scores[i][0]))
        parent, filename = os.path.split(input_image_path)
        save_name = os.path.join(parent, "output_" + filename)

        cv2.imwrite(save_name, image_raw)


    def infer_batch_host():
        pass

    def infer_batch_stack():
        pass

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

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


    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,x1,y1,x2,y2,conf,landmark_x1,landmark_y1,
            landmark_x2,landmark_y2,...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        cfg = cfg_mnet
        scale = torch.Tensor([640, 640, 640, 640])
        scale = scale.to(self.device)

        loc, landms, conf = output
        loc = torch.tensor(loc[3]).unsqueeze(0).to(self.device)
        conf = torch.tensor(conf[3]).unsqueeze(0).to(self.device)
        landms = torch.tensor(landms[3]).unsqueeze(0).to(self.device)
        print(loc.shape)
        print(conf.shape)
        print(landms.shape)

        priorbox = PriorBox(cfg, image_size=(640, 640))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([640, 640, 640, 640,
                               640, 640, 640, 640,
                               640, 640])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        bboxes = dets[:, :4]
        scores = dets[:, 4:]
        return bboxes, scores, landms


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
    # engine_file_path = "models/weights/retinaface_1.trt"
    engine_file_path = "models/weights/retinaface.trt"

    retinaface = Retinaface_trt(engine_file_path)
    input_image_paths = ["curve/images.jpeg"]
    for i in range(20):
        for input_image_path in input_image_paths:
            # create a new thread to do inference
            thread = myThread(retinaface.infer, [input_image_path])
            thread.start()
            thread.join()

    # destroy the instance
    retinaface.destroy()
