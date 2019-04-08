#!/usr/bin/env python
#-*- coding: utf-8 -*-
from extract_warped_flow import *
from test_minibatch import *
import cv2
import imageio as imo
import time
import collections
import numpy as np


def get_img_list_from_vid_reader(vid, reshape_size=(640, 360), MAX_N_FRAME=300):
    fps = int(round(vid.get_meta_data()['fps']))
    img_list = []

    for i in range(MAX_N_FRAME):
        num = i * fps
        try:
            img = vid.get_data(num)
        except:
            return np.array(img_list)

        img = cv2.resize(img, reshape_size).astype(np.float32)
        img_list.append(img)

    return np.array(img_list)


if __name__ == "__main__":
    video_path = "/home/shuo/Documents/MA/data/HealthCare/Resized/Videos_1/v_gloveson_12.MP4"
    # cap = cv2.VideoCapture(video_path)        #ersetzen mit 0 für webcam
    clip_size = 5
    rgb_folder = "/home/shuo/dense_flow/tools/rgbFrames"  #für was brauch ich diese ordner?
    flow_folder = "/home/shuo/dense_flow/tools/flowFrames"   #werden die Images der Webcam zwischen gespeichert
    reader = imo.get_reader(video_path)
    img_list = get_img_list_from_vid_reader(reader)
    print("lenght image list", len(img_list))
    cnt = 0
    frames = collections.deque()
    h, w, c = img_list[0].shape

    model = I3DAttModel(img_height=h,
                        img_width=w,
                        ckpt_name="HealthCare_twoStream_0.859_model-6000",
                        ckpt_dir="6-minibatch-att-twostream",
                        model_path="finetune-HealthCare-flow-5RGB5Flows",
                        dataset="HealthCare",
                        mode="twoStream",
                        batch_size=1,
                        clip_size=5,
                        crop_size=224,
                        class_num=6)

    while cnt < len(img_list):
        # ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frames = np.zeros((clip_size, h, w, c))
        # for i in range(clip_size):
        # 	ret, frame = cap.read()
        # 	frames[i] = frame
        if cnt < clip_size + 1:
            frames.append(img_list[cnt])
            cnt += 1
            continue
        elif cnt >= clip_size + 1:
            start_time = time.time()
            # ret, frame = cap.read()
            for j in range(clip_size + 1):
                cv2.imwrite(os.path.join(rgb_folder, "frame0000%02d.jpg" % j), frames[j])
            # cv2.imwrite(os.path.join(rgb_folder, "frame0000%02d.jpg" % (cnt + 1)))
            f = FlowExtractor(dev_id=0)
            c_flows = f.extract_warp_flow(h, w, clip_size)
            flow_frames = np.zeros((clip_size, h, w, 2))
            for j in range(clip_size):
                flow_frames[j, :, :, 0] = c_flows[2 * j, :, :]
                flow_frames[j, :, :, 1] = c_flows[2 * j + 1, :, :]
            # --------- load model ---------- #
            label_dict = {'0': "gloveson",
                          '1': "clean",
                          '2': "unpacking",
                          '3': "glovesoff",
                          '4': "disinfect",
                          '5': "other"}

            
            pred_prob, pred_class = model.pred(np.asarray(frames)[1:], flow_frames)
            mask = frames[-1]
            # write text on image "%s: %f" % (label_dict[str(pred_class)], pred_prob[pred_class])
            cv2.putText(mask, '%s' % label_dict[str(pred_class[0])], (h // 12, w // 12), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 69, 0))

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            cv2.imshow("Live", mask.astype('uint8'))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break            
            print("Duration %fs" % (time.time() - start_time))
            # save_optical_flow(flow_folder, flow_frames)
            frames.popleft()
            frames.append(img_list[cnt])
            cnt += 1
            continue