
# -*- coding: utf-8 -*-
from cgi import print_arguments
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.data.dataloader as DataLoader
from sklearn.metrics import f1_score
import sys
#from data import videoDataset
from fdp import FDP

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
    
sequence_loss = nn.MSELoss(reduction="mean")#SeqLoss(test_mode=True)

def train(args, model, train_dataset, test_dataset=None, train_log_file=None, test_log_file=None ,subject=""):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam([{"params":filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr, weight_decay=args.wdecay)
    '''
    optimizer = torch.optim.SGD([{"params":filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr, momentum=args.momentum,
                                weight_decay=args.wdecay, nesterov=True)
    '''
    train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    total_steps = 0
    keep_training = True
    epoch = 0
    final_acc = 0

    while keep_training:
        torch.cuda.empty_cache()
        epoch += 1
        args.test_mode = False
        totalsamples = 0
        correct_samples = 0
        acc = 0

        for i, item in enumerate(train_dataloader):
            print('-----epoch:{}  steps:{}/{}-----'.format(epoch, total_steps, args.num_steps))
            video, dyimg , label = item
            dyimg = dyimg/255.0
            print(video.size())
            #dy_loss = torch.zeros(1).requires_grad_(True)
            optimizer.zero_grad()
            pred_mer, pred_dyimg = model(video.to(device))
            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer, label.to(device))
            _, pred = torch.max(pred_mer, dim=1)

            print('label:{} pred:{}'.format(label, pred))

            dy_loss = sequence_loss(pred_dyimg.to(torch.float32), dyimg.to(torch.float32).to(device))

            print("ME_LOSS:",ME_loss ,"DY_LOSS:", dy_loss)

            final_loss =dy_loss.to(torch.float32).cpu() * args.dy_weight + ME_loss.to(torch.float32).cpu()* args.mer_weight

            final_loss.backward()
            optimizer.step()

            batch_correct_samples = pred.cpu().eq(label).sum()
            correct_samples += pred.cpu().eq(label).sum()
            totalsamples += len(label)
            batch_acc = batch_correct_samples / len(label)
            acc = correct_samples / totalsamples
            print("batch_acc:{}%".format(batch_acc * 100))
            print("acc:{}%".format(acc * 100))


            train_log_file.writelines('-----epoch:{}  steps:{}/{}-----\n'.format(epoch, total_steps, args.num_steps))
            train_log_file.writelines(
                'DY loss:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(dy_loss, ME_loss, final_loss))
            train_log_file.writelines('batch acc:{}\t\tacc:{}\n'.format(batch_acc * 100, acc * 100))
            total_steps += 1

            if total_steps > args.num_steps:
                keep_training = False
                break

        print("epoch average acc:{}%".format(acc * 100))
        print('=========================')
        train_log_file.writelines('epoch average acc:{}%\n'.format(acc * 100))
        train_log_file.writelines('=========================\n')
        acc = evaluate(args, model, epoch=epoch, test_dataset=test_dataset, test_log_file=test_log_file)
        if acc > final_acc:
            torch.save(model.state_dict(), args.save_path)
            final_acc = acc
        if total_steps % 2000 ==0:
            torch.save(model.state_dict(), 'saved_models/{}_RPMER_{}_{}.pth'.format(args.version,str(total_steps),subject))

def evaluate(args, model, epoch, test_dataset, test_log_file):

    args.test_mode = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    totalsamples = 0
    correct_samples = 0
    
    pred_list = []
    label_list = []

    global confusion_matrix 
    confusion_matrix = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]


    test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=48)

    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            print('-----epoch:{}  batch:{}-----'.format(epoch, i))

            video, dyimg, label = item
            dyimg = dyimg/255.0

            pred_mer,pred_dyimg = model(video.to(device))

            pred_mer = F.log_softmax(pred_mer, dim=1)
            ME_loss = F.nll_loss(pred_mer, label.to(device))
            _, pred = torch.max(pred_mer, dim=1)
            pred_list.extend(pred.cpu().numpy().tolist())

            label_list.extend(label.numpy().tolist())    

            print('label:{} \n pred:{}'.format(label, pred))
            dy_loss = sequence_loss(pred_dyimg.to(torch.float32), dyimg.to(torch.float32).to(device))

        correct_samples += cal_corr(label_list, pred_list)
        totalsamples += len(label_list)
        acc = correct_samples * 100.0 / totalsamples
        weighted_f1_score = f1_score(label_list, pred_list, average="weighted") * 100
        print('-----epoch:{}-----'.format(epoch))
        print("acc:{}%".format(acc))
        print("weighted f1 score:{}".format(weighted_f1_score))

        test_log_file.writelines('\n-----epoch:{}-----\n'.format(epoch))
        test_log_file.writelines('acc:{}\t\tweighted_f1:{}\n'.format(acc, weighted_f1_score))
        final_loss =dy_loss.to(torch.float32).cpu() * args.dy_weight + ME_loss.to(torch.float32).cpu()* args.mer_weight

        test_log_file.writelines('DY LOSS:{}\t\tME loss:{}\t\tFinal loss:{}\n'.format(dy_loss, ME_loss, final_loss))
        test_log_file.writelines('confusion_matrix:\n{}\n{}\n{}\n{}\n{}\n'.format(confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3],confusion_matrix[4]))
    
    print(confusion_matrix)
    return acc

def cal_corr(label_list, pred_list):
    corr = 0
    for (a, b) in zip(label_list, pred_list):
        confusion_matrix[a][b]+=1
        if a == b:
            corr += 1
    return corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--wdecay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dy_weight', type=float, default = 100)
    parser.add_argument('--mer_weight', type=float, default = 1)
    parser.add_argument('--dataset', type=str, default="SAMM")
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--version', default='test')
    parser.add_argument('--net_test',action='store_true')
    args = parser.parse_args()

    if args.dataset == "CASME2":
        LOSO = ['17', '26', '16', '09', '05', '24', '02', '13', '04', '23', '11', '12', '08', '14', '03', '19', '01', '10', '20', '21', '22', '15', '06', '25', '07']

    if args.dataset == "SAMM":
        LOSO =['006','007','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','025','026','028','030','031','032','033','034','035','036','037']

    test_log_file = open('log/' + args.version + '_test_log.txt', 'w')
    train_log_file = open('log/' + args.version + '_train_log.txt', 'w')

    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
        train_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
        test_log_file.writelines('%s: %s\n' % (k, vars(args)[k]))
    train_log_file.writelines('----------args----------\n')
    test_log_file.writelines('----------args----------\n')

    #model = RP_MER()
    if args.save_path is None:
            args.save_path = './{}.pth'.format(args.version)

    for sub in range(len(LOSO)):
        model = FDP()
        subject = LOSO[sub]
        print('Loading dataset...')
        train_dataset = torch.load('dataset/FDP_'+args.dataset+'_train_'+subject+'subject_5cls.pth')
        test_dataset = torch.load('dataset/FDP_'+args.dataset+'_test_'+subject+'subject_5cls.pth')

        train_dataset_size = train_dataset.__len__()
        test_dataset_size = test_dataset.__len__()
        train_log_file.writelines('train_dataset_size:' + str(train_dataset_size))
        train_log_file.writelines('test_dataset_size:'+ str(test_dataset_size))
        print('train_dataset.size:{}'.format(len(train_dataset)))
        train_log_file.writelines('LOSO ' +subject+'\n')
        test_log_file.writelines('LOSO '+subject+'\n')
        final_acc=train(args=args, model=model, train_dataset=train_dataset, test_dataset=test_dataset,train_log_file=train_log_file, test_log_file=test_log_file,subject = subject)
        test_log_file.writelines('LOSO '+subject+' best_acc:'+str(final_acc)+'\n')
