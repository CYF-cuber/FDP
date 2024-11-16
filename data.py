# -*- coding: utf-8 -*-
from cgi import print_arguments
import os
import numpy as np
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader
from torchsummary import summary
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
import random
import dlib
import math

import sys


def align_face(img, img_land, box_enlarge, img_size):

    leftEye0 = (img_land[2 * 37] + img_land[2 * 38] + img_land[2 * 39] + img_land[2 * 40] + img_land[2 * 41] +
                img_land[2 * 36]) / 6.0
    leftEye1 = (img_land[2 * 37 + 1] + img_land[2 * 38 + 1] + img_land[2 * 39 + 1] + img_land[2 * 40 + 1] +
                img_land[2 * 41 + 1] + img_land[2 * 36 + 1]) / 6.0
    rightEye0 = (img_land[2 * 43] + img_land[2 * 44] + img_land[2 * 45] + img_land[2 * 46] + img_land[2 * 47] +
                 img_land[2 * 42]) / 6.0
    rightEye1 = (img_land[2 * 43 + 1] + img_land[2 * 44 + 1] + img_land[2 * 45 + 1] + img_land[2 * 46 + 1] +
                 img_land[2 * 47 + 1] + img_land[2 * 42 + 1]) / 6.0
    deltaX = float(rightEye0 - leftEye0)
    deltaY = float(rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    #print("pupil distance:",l)
    #print(deltaX,deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])

    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 30], img_land[2 * 30 + 1], 1],
                   [img_land[2 * 48], img_land[2 * 48 + 1], 1], [img_land[2 * 54], img_land[2 * 54 + 1], 1]])
    
    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1
    #print((mat[0:2, :]))
    #print(mat1)
    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))

    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

def img_pre_dlib(detector,predictor,img_path,box_enlarge=2.0,img_size=64):
    img = cv2.imread(img_path)#[:,80:560]

    img_dlib = dlib.load_rgb_image(img_path)
    dets = detector(img_dlib, 1)
    shape = predictor(img_dlib, dets[0])
    ldm = np.matrix([[p.x, p.y] for p in shape.parts()])
    ldm=ldm.reshape(136,1)

    aligned_img, new_land = align_face(img, ldm, box_enlarge, img_size)
    return aligned_img

class videoDataset(Dataset):
    def __init__(self, video_path):
        self.video_list = []
        self.video_labels = []

        for video in range(len(video_path)):
            self.load_class_video(video_path[video], class_num = video)
        
        self.video_list = np.asarray(self.video_list)
        print(self.video_list.shape)

        self.training_samples = len(self.video_list)

        #self.video_list = np.expand_dims(self.video_list, axis=1).astype('float32')
        self.video_list = self.video_list.astype(np.float32)
        print(self.video_list)
        print(self.video_labels)
        
        print(self.video_list.shape)
        self.video_list = self.video_list - np.mean(self.video_list)
        self.video_list = self.video_list / np.max(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):
        return self.video_list[item], self.video_labels[item]

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def video_length_judge(self, framelist):
        video_len = len(framelist)
        if video_len <8 :
            return False
        else:
            return True

    def load_video(self, video_path, framelist):

        video_len = len(framelist)

        sample_time = video_len // 8
        frames = []

        for i in range(8):
            img_path = video_path + '/' + framelist[i * sample_time]
            align_img = img_pre_dlib(detector,predictor,img_path)
            print(align_img.shape)
            #cv2.imwrite("/mnt/data/experiments/micro-expression-recognition-master/Frequency_MER/img/test_ali.jpg",align_img)

            frames.append(align_img)

        frames = np.asarray(frames)
        print(frames.shape)
        
        return frames

    def load_class_video(self, video_path,class_num = None):

        directorylisting = video_path
        for video in directorylisting:

            videopath = video_list[class_num] + video
  
            print(videopath)
            framelist = os.listdir(videopath)
            
            if "EP" in video:
                framelist.sort(key=lambda x: int(x.split('img')[1].split('.jpg')[0]))
            elif 's' in video:
                framelist.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0]))
            else:
                framelist.sort(key=lambda x: int(x.split('.')[0]))

            if self.video_length_judge( framelist) is False:
                continue

            videoarray= self.load_video(videopath, framelist)
            
            videoarray = np.rollaxis(videoarray,3,0)
            print(videoarray.shape)

            if len(videoarray) <= 0:
                print("video invalid!")
                continue
            self.video_list.append(videoarray)
            # 添加标签
            self.video_labels.append(class_num % 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="SAMM")
    parser.add_argument('--net_test',action='store_true')
    args = parser.parse_args()
    
    if args.dataset == "CASME2":
        surprisepath_c = '../CASME2_data_3/surprise/'
        positivepath_c = '../CASME2_data_3/positive/'
        negativepath_c = '../CASME2_data_3/negative/'
        video_list = [surprisepath_c, positivepath_c, negativepath_c]
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']
        #[7, 8, 10, 12, 13, 17, 18, 19, 20, 24]
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '23', '12',  '03', '19', '01', '15', '06', '25']


    if args.dataset == "SAMM":
        surprisepath_s = '../SAMM_data_3/surprise/'
        positivepath_s = '../SAMM_data_3/positive/'
        negativepath_s = '../SAMM_data_3/negative/'
        video_list = [surprisepath_s, positivepath_s, negativepath_s]
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','026','028','030','031','032','033','034','035','036','037']
        #[3, 6, 12,  14,  16, 17, 18, 19,  20,  21, 22,  24,  26,  27]
        LOSO =['006','007','009','011','012','014','015','016','017','018','020','022','033','035']


    '''##################################################################################
    If you want to create 5-cls dataset, use Line 205-223 instead of 182-199 and change Line 174 to "            self.video_labels.append(class_num % 5)" and Line 233 to "        test_list = [[],[],[],[],[]]"
    '''
    # if args.dataset == "CASME2":
    #     surprise_path = '../CASME2_data_5/surprise/'
    #     happiness_path = '../CASME2_data_5/happiness/'
    #     disgust_path = '../CASME2_data_5/disgust/'
    #     repression_path = '../CASME2_data_5/repression/'
    #     others_path = '../CASME2_data_5/others/'
    #     video_list = [surprise_path , happiness_path, disgust_path , repression_path , others_path]
    #     LOSO = ['01', '17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '10', '20', '21', '22', '15', '06', '25', '07']
    #     #LOSO = ['25', '07']
    # if args.dataset == "SAMM":
    #     surprise_path = '../SAMM_data_5/surprise/'
    #     happiness_path = '../SAMM_data_5/happiness/'
    #     anger_path = '../SAMM_data_5/anger/'
    #     contempt_path = '../SAMM_data_5/contempt/'
    #     others_path = '../SAMM_data_5/others/'
    #     video_list = [surprise_path , happiness_path, anger_path , contempt_path , others_path]
    #     #LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','028','030','031','032','033','034','035','036','037']
    #     # del '024'
    #     LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','025','026','028','030','031','032','033','034','035','036','037']


    videos = [os.listdir(i) for i in video_list]
    
    one_cls_sub = []
    
    for sub in range(len(LOSO)):

        subject = LOSO[sub]
        test_list = [[],[],[]]
        for cla in range(len(videos)):
            class_video = videos[cla]
            for v in class_video:
                if v.split('_')[0] == subject:
                    test_list[cla].append(v)

        print(test_list)
        
        blank_count =0  
        for c in test_list:
            if len(c) == 0:
                blank_count = blank_count+1
            if blank_count == 2:
                one_cls_sub.append(sub)
        if args.net_test:
            test_list = [c[0:1] for c in test_list]     
        test_dataset =  videoDataset(test_list)
        torch.save(test_dataset, './mer_dataset/FDP_'+args.dataset+'_3cls_sub'+subject+'.pth')
