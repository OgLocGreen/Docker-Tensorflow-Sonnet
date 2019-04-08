#!/usr/bin/env python
#-*- coding: utf-8 -*-
import sys
import tensorflow as tf

sys.path.append('/home/shuo/dense_flow/build/')

import os
from libpydenseflow import TVL1FlowExtractor, TVL1WarpFlowExtractor
import numpy as np

class FlowExtractor(object):

    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1WarpFlowExtractor(bound)

    # def extract_warp_flow(self, frame_list, new_size=None):
    def extract_warp_flow(self, img_height, img_width, clip_size, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        # frame_size = frame_list[0].shape[:2]
        # rst = self._et.extract_warp_flow([x.tostring() for x in frame_list], frame_size[1], frame_size[0])
        # rst = self._et.extract_warp_flow([x for x in frame_list], 456, 256)
        # rst = self._et.extract_warp_flow(frame_list[0].tostring(), frame_size[1], frame_size[0])
        rst = self._et.extract_warp_flow(img_width, img_height, clip_size)
        # rst = self._et.extract_warp_flow([np.chararray(x, unicode=True) for x in frame_list], frame_size[1], frame_size[0])
        n_out = len(rst)
        if new_size is None:
            ret = np.zeros((n_out*2, img_height, img_width))
            for i in range(n_out):
                ret[2*i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape([img_height, img_width])
                ret[2*i+1, :] = np.fromstring(rst[i][1], dtype='uint8').reshape([img_height, img_width])
        else:
            import cv2
            ret = np.zeros((n_out*2, new_size[1], new_size[0]))
            for i in range(n_out):
                ret[2*i, :] = cv2.resize(np.fromstring(rst[i][0], dtype='uint8').reshape([img_height, img_width]), new_size)
                ret[2*i+1, :] = cv2.resize(np.fromstring(rst[i][1], dtype='uint8').reshape([img_height, img_width]), new_size)

        return ret

def save_optical_flow(output_folder, flow_frames):
    try:
        os.mkdir(output_folder)
    except OSError:
        pass
    nframes = len(flow_frames) // 2
    for i in range(nframes):
        out_x = '{0}/x_{1:04d}.jpg'.format(output_folder, i+1)
        out_y = '{0}/y_{1:04d}.jpg'.format(output_folder, i+1)
        cv2.imwrite(out_x, flow_frames[2*i])
        cv2.imwrite(out_y, flow_frames[2*i+1])



if __name__ == "__main__":
    import cv2
    import time
    video_path = "/home/shuo/Videos/trees.avi"
    # cap = cv2.VideoCapture(video_path)
    clip_size = 16
    rgb_folder = "/home/shuo/dense_flow/tools/rgbFrames"
    flow_folder = "/home/shuo/dense_flow/tools/flowFrames"

    while True:
        # ret, frame = cap.read()

        frame = cv2.imread(os.path.join(rgb_folder, "frame000001.jpg"))
        h, w, c = frame.shape
        start_time = time.time()
        # ret, frame = cap.read()
        # cv2.imwrite(os.path.join(rgb_folder, "frame0000%02d.jpg" % (cnt + 1)))
        f = FlowExtractor(dev_id=0)
        flow_frames = f.extract_warp_flow(h, w, clip_size)

        # --------- load model ---------- #
        print("Duration %fs" % (time.time() - start_time))
        save_optical_flow(flow_folder, flow_frames)
