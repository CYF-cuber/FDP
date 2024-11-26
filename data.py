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

image_rows, image_columns, image_depth = 64, 64, 8
VIDEO_LENGTH = 8

def get_dynamic_image(frames, normalized=True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2]                                                        # num_channels = 3
    channel_frames = _get_channel_frames(frames, num_channels)                               # channel_frames.shape = (3, 51, 316, 274, 1)
     
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames] # channel.shape = (51, 316, 274, 1) 
                                                                                             # channel_dynamic_images = (3, 316, 274, 1) 
    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image

def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for channel in range(num_channels)] # frames = [[] [] []]
    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)):                       # b, g ,r =cv2.split(frame) channel_frames=[], channel.shape = (316,274)
            channel_frames.append(channel.reshape((channel.shape[0], channel.shape[1], 1))) # python2                                       #  =  (1,316,274,1)
    for i in range(len(frames)):                                                            # frames.shape = (3, 51, 316, 274, 1)
        frames[i] = np.array(frames[i])
    return frames

def _compute_dynamic_image(frames):
    """ Inspired by https://github.com/hbilen/dynamic-image-nets """
    num_frames, h, w, depth = frames.shape                                              # = 51 316 274 1
    y   = np.zeros((num_frames, h, w, depth))                                           # = 51*316*274*1
    ids = np.ones(num_frames)                                                           # = 51*1                              
    fw  = np.zeros(num_frames)                                                          # shape = (51,)    fw is the score (frame weight)

    for n in range(num_frames):    
        cumulative_indices = np.array(range(n, num_frames)) + 1
        fw[n] = np.sum(((2*cumulative_indices) - num_frames-1) / cumulative_indices)      # python 3 & 2 # fw = [-4/3, 2/3, 2/3 ] when numframes = 3 

    for v in range(int(np.max(ids))):                                # v = 0 all the time ?
        indv = np.array(np.where(ids == v+1))                        # shape = (1,51)  (0,1,...,50)
        a1 = frames[indv, :, :, :]                                   # a1 = frames ???   a1.shape=(1,3,224,224,1)
        a2 = np.reshape(fw, (indv.shape[1], 1, 1, 1))                # a2.shape = (3,1,1,1)  a2 is the b in paper?
        a3 = a1 * a2                                                 # (1,3,h,w,1) * (3,1,1,1) = (1,3,h,w,1)
        y = np.sum(a3[0], axis=0)                                    # y.shape = (h, w, 1)
    return y 

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

    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))

    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def img_pre_dlib(detector,predictor,img_path,box_enlarge=2.5,img_size=64):
    img = cv2.imread(img_path)#[:,80:560]

    img_dlib = dlib.load_rgb_image(img_path)
    dets = detector(img_dlib, 1)
    shape = predictor(img_dlib, dets[0])
    ldm = np.matrix([[p.x, p.y] for p in shape.parts()])
    ldm=ldm.reshape(136,1)

    aligned_img, new_land = align_face(img, ldm, box_enlarge, img_size)
    new_land = new_land.reshape(1,136)
    return aligned_img , new_land

def image_pro(img):
    normalize = transforms.Normalize(mean=[0.5],std=[0.5])

    transform_1 = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    transform_2 = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        normalize
    ])
    gray_img = transform_1(img)
    norm_gray_img = transform_2(img)
    #print(norm_img.size())
    return gray_img, norm_gray_img

class videoDataset(Dataset):
    def __init__(self, video_path, load_data=False, mode_train =True):
        self.video_list = []
        self.dyimg_list = []
        self.video_labels = []
        self.mode_train = mode_train

        for video in range(len(video_path)):
            self.load_class_video(video_path[video], load=load_data , mode_train =self.mode_train , class_num = video)
        self.video_list = np.asarray(self.video_list)

        self.dyimg_list = np.asarray(self.dyimg_list)
        self.training_samples = len(self.video_list)

        self.video_list = np.expand_dims(self.video_list, axis=1).astype('float32')

        self.video_list = self.video_list - np.mean(self.video_list)
        self.video_list = self.video_list / np.max(self.video_list)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, item):
        return self.video_list[item], self.dyimg_list[item], self.video_labels[item]


    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def video_length_judge(self, video_path, framelist):
        video_len = len(framelist)
        if video_len <VIDEO_LENGTH :
            print(video_path + " is < 8 frames!")
            return False
        else:
            return True

    def load_video(self, video_path, framelist, load_frame_data=False, video_name='',mode_train=True):

        video_len = len(framelist)

        sample_time = video_len // VIDEO_LENGTH
        frames = []
        frames_dyimg = []

        for i in range(VIDEO_LENGTH):
            img_path = video_path + '/' + framelist[i * sample_time]
            align_img,ldm = img_pre_dlib(detector,predictor,img_path)
            #print(align_img.shape)
            image =  Image.fromarray(cv2.cvtColor(align_img,cv2.COLOR_BGR2RGB))
            #print(type(image))
            gray_img, norm_gray_img = image_pro(image)
            frames_dyimg.append(gray_img.numpy())
            frames.append(norm_gray_img.numpy())

        frames_dyimg = np.asarray(frames_dyimg).reshape(8,64,64,1)

        dy_img = get_dynamic_image(frames_dyimg)

        frames = np.asarray(frames)

        return frames, dy_img

    def load_class_video(self, video_path, load=False,mode_train = True,class_num = None):
        #directorylisting = os.listdir(video_path)
        directorylisting = video_path
        for video in directorylisting:

            videopath = video_list[class_num] + video
            #print(videopath)
            '''
            if not os.path.exists(videopath):
                videopath = video_list[class_num+3] +video
            '''
            print(videopath)
            framelist = os.listdir(videopath)

            #framelist.sort(key=lambda x: int(x.split('img')[1].split('.jpg')[0]))
            
            if "EP" in video:
                framelist.sort(key=lambda x: int(x.split('img')[1].split('.jpg')[0]))
            elif 's' in video:
                framelist.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0]))
            else:
                framelist.sort(key=lambda x: int(x.split('.')[0]))

            if self.video_length_judge(videopath, framelist) is False:
                continue

            if mode_train:
                videoarray, dy_img = self.load_video(videopath, framelist, load_frame_data=False, video_name=video,mode_train=True)
            else:
                videoarray, dy_img = self.load_video(videopath, framelist, load_frame_data=False, video_name=video,mode_train=False)

            if len(videoarray) <= 0:
                print("video invalid!")
                continue
            self.video_list.append(videoarray)
            self.dyimg_list.append(dy_img)
            self.video_labels.append(class_num % 5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="SAMM")
    parser.add_argument('--net_test',action='store_true')
    args = parser.parse_args()

    
    if args.dataset == "CASME2":
        surprise_path = './CASME2_data_5/surprise/'
        happiness_path = './CASME2_data_5/happiness/'
        disgust_path = './CASME2_data_5/disgust/'
        repression_path = './CASME2_data_5/repression/'
        others_path = './CASME2_data_5/others/'
        video_list = [surprise_path , happiness_path, disgust_path , repression_path , others_path]
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']
    if args.dataset == "SAMM":
        surprise_path = './SAMM_data_5/surprise/'
        happiness_path = './SAMM_data_5/happiness/'
        anger_path = './SAMM_data_5/anger/'
        contempt_path = './SAMM_data_5/contempt/'
        others_path = './SAMM_data_5/others/'
        video_list = [surprise_path , happiness_path, anger_path , contempt_path , others_path]
        #LOSO =['006','007']
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','025','026','028','030','031','032','033','034','035','036','037']



    test_log_file = open('log/dataset_test_log.txt', 'w')
    train_log_file = open('log/dataset_train_log.txt', 'w')

    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
        train_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
        test_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')

    videos = [os.listdir(i) for i in video_list]

    for sub in range(len(LOSO)):
        subject = LOSO[sub]
        test_list = [[],[],[],[],[]]
        for cla in range(len(videos)):
            class_video = videos[cla]
            for v in class_video:
                if v.split('_')[0] == subject:
                    test_list[cla].append(v)

        print(test_list)
        train_list = [[x for x in videos[y] if x not in test_list[y]] for y in [0,1,2,3,4]]
        print(train_list)
        if args.net_test:
            test_list = [c[0:1] for c in test_list]
            #print(test_list)
            train_list = [d[0:1] for d in train_list]
        test_dataset =  videoDataset(test_list, load_data=True,mode_train=False)
        train_dataset = videoDataset(train_list, load_data=True,mode_train=True)

        train_dataset_size = train_dataset.__len__()
        test_dataset_size = test_dataset.__len__()
        train_log_file.writelines('train_dataset_size:' + str(train_dataset_size))
        train_log_file.writelines('test_dataset_size:'+ str(test_dataset_size))
        print('Generating dataset...')
        print('train_dataset.size:{}'.format(train_dataset_size)+'\n')
        print('test_dataset.size:{}'.format(test_dataset_size)+'\n')


        print('Saving dataset...')
        torch.save(train_dataset, 'dataset/FDP_'+args.dataset+'_train_'+subject+'subject_5cls.pth')
        torch.save(test_dataset, 'dataset/FDP_'+args.dataset+'_test_'+subject+'subject_5cls.pth')
        
        train_log_file.writelines('LOSO ' +subject+'\n')# %s\n' % (str(i+1)))
        test_log_file.writelines('LOSO '+subject+'\n')# %s\n' % (str(i+1)))
