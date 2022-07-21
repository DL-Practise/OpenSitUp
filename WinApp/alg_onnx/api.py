import sys
import argparse
import os
import time
import cv2
import torch
import torchvision
import numpy as np
import onnxruntime

def situp_preproc(cv_img, resize=(224,224), mean=[103.53,116.28,123.675], std=[57.375,57.12,58.395]):
    img = cv2.resize(cv_img, resize)
    img = img - mean
    img = img / std
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = np.expand_dims(img.transpose(2,0,1), axis=0)
    return img

class SitupDet():
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(cur_dir, './models/model_24.onnx')
        # 1. 根据onnx模型，创建onnx的session
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.onnx_input_info = self.onnx_session.get_inputs()[0]
        self.onnx_output_info = self.onnx_session.get_outputs()[0]

    def infer(self, cv_img):
        img_ori = cv_img.copy()
        h, w, c = cv_img.shape
        img = situp_preproc(cv_img)
        input_feed = {self.onnx_input_info.name: img}
        # 2. 根据onnx_session,进行预测推理，输入图片，得出结果
        preds = self.onnx_session.run([self.onnx_output_info.name], input_feed=input_feed)
        preds = preds[0]

        featuremap1 = preds[0][0]
        featuremap2 = preds[0][1]
        featuremap3 = preds[0][2]

        max1 = np.unravel_index(np.argmax(featuremap1), featuremap1.shape)
        max2 = np.unravel_index(np.argmax(featuremap2), featuremap2.shape)
        max3 = np.unravel_index(np.argmax(featuremap3), featuremap3.shape)

        norm_pos1 = [(max1[0] + 0.5) / featuremap1.shape[0], (max1[1] + 0.5) / featuremap1.shape[1]]
        norm_pos2 = [(max2[0] + 0.5) / featuremap2.shape[0], (max2[1] + 0.5) / featuremap2.shape[1]]
        norm_pos3 = [(max3[0] + 0.5) / featuremap3.shape[0], (max3[1] + 0.5) / featuremap3.shape[1]]
        abs_pos1 = (int(norm_pos1[1] * w), int(norm_pos1[0] * h))
        abs_pos2 = (int(norm_pos2[1] * w), int(norm_pos2[0] * h))
        abs_pos3 = (int(norm_pos3[1] * w), int(norm_pos3[0] * h))

        cv2.line(img_ori, abs_pos1, abs_pos3, (0, 250, 250), 3)
        cv2.line(img_ori, abs_pos2, abs_pos3, (0, 250, 250), 3)
        cv2.circle(img_ori, abs_pos1, 10, (0, 0, 255), -1)
        cv2.circle(img_ori, abs_pos2, 10, (0, 255, 0), -1)
        cv2.circle(img_ori, abs_pos3, 10, (255, 0, 0), -1)

        return img_ori, [norm_pos1, norm_pos2, norm_pos3]
        

if __name__ == '__main__':

    #############################################################
    img_path = 'C:\\Users\\Administrator\\Desktop\\2.jpg'
    #############################################################
    
    situp_det = SitupDet()
    now_dir = os.path.dirname(__file__)

    img_list = []
    if os.path.isdir(img_path):
        for file in os.listdir(img_path):
            if str(file).endswith('jpg'):
                img_list.append(os.path.join(img_path, file))
    else:
        img_list.append(img_path)

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        img_result, results = situp_det.infer(img)
        #save_img_ori_path = os.path.join(now_dir, 'save/%d_ori.jpg'%(i))
        #save_img_result_path = os.path.join(now_dir,'save/%d_result.jpg'%(i))
        #cv2.imwrite(save_img_ori_path, img)
        #cv2.imwrite(save_img_result_path, img_result)
        #plt.title(os.path.basename(img_path))
        #plt.imshow(img[:,:,[2,1,0]])
        #plt.show()
