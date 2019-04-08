from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import i3d
from inference import *
import random
import cv2
import numpy as np


class I3DAttModel(object):

    def __init__(self, img_height, img_width, ckpt_name, ckpt_dir, model_path, dataset, mode, batch_size, clip_size, crop_size, class_num):
        self.ckpt_name = ckpt_name
        self.ckpt_dir = ckpt_dir
        self.model_path = model_path
        self.dataset = dataset
        self.mode = mode
        self.batch_size = batch_size
        self.clip_size = clip_size
        self.crop_size = crop_size
        self.class_num = class_num
        self.h = img_height
        self.w = img_width
        self.log_root = "./output"
        test_ckpt = os.path.join(self.log_root, self.ckpt_dir, self.model_path, self.ckpt_name)


        self.image_holder_rgb = tf.placeholder(tf.float32, [self.batch_size, self.clip_size, self.crop_size, self.crop_size, 3])
        self.image_holder_flow = tf.placeholder(tf.float32, [self.batch_size, self.clip_size, self.crop_size, self.crop_size, 2])
        self.label_holder = tf.placeholder(tf.int32, [self.batch_size])
        self.dropout_holder = tf.placeholder(tf.float32)
        self.is_train_holder = tf.placeholder(tf.bool)

        fc_out, weights_rgb_1, logits_rgb = att_model(self.image_holder_rgb, self.image_holder_flow, self.dropout_holder, False,
                                                      self.class_num)

        self.softmax = tf.nn.softmax(fc_out)

        variable_map_rgb, variable_map_flow, variable_map_twostream = {}, {}, {}
        for variable in tf.global_variables():
            tmp = variable.name.split('/')
            if tmp[0] == 'RGB':
                variable_map_rgb[variable.name.replace(':0', '')] = variable
            elif tmp[0] == 'Flow':
                variable_map_flow[variable.name.replace(':0', '')] = variable
            elif tmp[0] == 'TwoStream':
                variable_map_twostream[variable.name.replace(':0', '')] = variable
        variable_map_twostream.update(variable_map_flow)
        variable_map_twostream.update(variable_map_rgb)

        # saver = tf.train.Saver(var_list=variable_map_twostream, reshape=True)  # max_to_keep: maximum number of recent checkpoints to keep
        saver = tf.train.Saver()
        self.sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, test_ckpt)


    def get_batch(self, rgb_imgs, flow_imgs):
        """
        rgb_frames: (5, h ,w, 3) nparray
        flow_frames: (10, h, w, 2) 
        flow_frames[2*i]: x
        flow_frames[2*i+1]: y
        """
       
        rgb_imgs_reshaped = [[] for _ in range(self.clip_size)]
        flow_imgs_reshaped = [[] for _ in range(self.clip_size)]
        if self.w > self.h:
            scale = float(256) / float(self.h)
            crop_x = random.randint(0, int(256 - self.crop_size))
            crop_y = random.randint(0, int(int(self.w * scale + 1) - self.crop_size))
            for i in range(self.clip_size):
                tmp1 = cv2.resize(rgb_imgs[i], (int(self.w * scale + 1), 256)).astype(np.float32)
                tmp2 = cv2.resize(flow_imgs[i], (int(self.w * scale + 1), 256)).astype(np.float32)
                rgb_imgs_reshaped[i] = tmp1[crop_x: (crop_x + self.crop_size), crop_y:(crop_y + self.crop_size), :]
                flow_imgs_reshaped[i] = tmp2[crop_x: (crop_x + self.crop_size), crop_y:(crop_y + self.crop_size), :]
                
        else:
            scale = float(256) / float(self.w)
            crop_x = random.randint(0, int(int(self.h * scale + 1) - self.crop_size))
            crop_y = random.randint(0, int(256 - self.crop_size))
            for i in range(self.clip_size):
                tmp1 = cv2.resize(rgb_imgs[i], (256, int(self.h * scale + 1))).astype(np.float32)
                tmp2 = cv2.resize(flow_imgs[i], (256, int(self.h * scale + 1))).astype(np.float32)

                rgb_imgs_reshaped[i] = tmp1[crop_x: (crop_x + self.crop_size), crop_y:(crop_y + self.crop_size), :]
                flow_imgs_reshaped[i] = tmp2[crop_x: (crop_x + self.crop_size), crop_y:(crop_y + self.crop_size), :]


        rgb_input = np.array(rgb_imgs_reshaped).reshape((self.batch_size, self.clip_size, self.crop_size, self.crop_size, 3)) / 255.
        flow_input = np.array(flow_imgs_reshaped).reshape((self.batch_size, self.clip_size, self.crop_size, self.crop_size, 2)) / 255.

        return rgb_input, flow_input


    def pred(self, rgb_imgs, flow_imgs):

        rgb_imgs_out, flow_imgs_out = self.get_batch(rgb_imgs, flow_imgs)

        pred_prob = self.sess.run(self.softmax, feed_dict={
                                                self.image_holder_rgb: rgb_imgs_out,
                                                self.image_holder_flow: flow_imgs_out,
                                                self.label_holder: np.ones(1, dtype=np.int64),
                                                self.dropout_holder: 1.0,
                                                self.is_train_holder: False})

        pred_class = np.argmax(pred_prob, axis=-1)

        return pred_prob, pred_class 


if __name__ == '__main__':
    print("Hey, you should not see this:)")