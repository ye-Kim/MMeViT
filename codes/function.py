import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor 
import math 
import numpy as np
from collections import OrderedDict

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torchsummary 

import pickle as pkl
import numpy as np
import glob
import gc
import os
import yaml 

import logging 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from imu.model import *
from imu.function import imuDataset 
from imu.preprocessing import many_sliding_window, many_sliding_window_all # preprocessing 불러오기

from skeleton.function import Discriminator, Feature3D_bf
from skeleton.multi_feature import heatman_Data

import sys 
sys.path.insert(0, './modeltest-results')

### mlp_head ###

class mlp_head(nn.Module):
    def __init__(self, mlp_dim, num_classes):
        super().__init__()

        self.latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.latent(x) 
        x = self.mlp_head(x)

        return x


### skeleton ###

class SkeletonWrapper():
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.config_file['input']['skeleton']
        self.config_model = self.config_file['model']['skeleton']
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mlp = mlp_head(self.config_model['data_emb_size'], self.config_file['n_classes']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def load_Dataset(self, train=False):
        if train:
            dataset = heatman_Data(heat_dir =self.config_data['heat_dir'],
                                label_csv=self.config_data['train_dir'], ceiling=self.config_data['ceiling'], mode=self.config_data['mode'])
        else:
            dataset = heatman_Data(heat_dir =self.config_data['heat_dir'],
                                label_csv=self.config_data['label_dir'], ceiling=self.config_data['ceiling'], mode=self.config_data['mode'])
        return dataset
    
    

    def load_Model(self):
        model = Discriminator(in_channels=self.config_model['in_channels'], patch_size=self.config_model['patch_size'], 
                                    data_emb_size=self.config_model['data_emb_size'], label_emb_size=self.config_model['label_emb_size'], 
                                    seq_length = self.config_model['seq_length']*128, depth=self.config_model['depth'], n_classes=self.config_file['n_classes'])
        param = torch.load(self.config_model['dir'])
        delList= self.config_model['delList']

        for i in delList:
            param.pop(i, None)
        model.load_state_dict(param)
       
        return model
    
    def load_Model_mlp(self):
        model = Discriminator(in_channels=self.config_model['in_channels'], patch_size=self.config_model['patch_size'], 
                                    data_emb_size=self.config_model['data_emb_size'], label_emb_size=self.config_model['label_emb_size'], 
                                    seq_length = self.config_model['seq_length']*128, depth=self.config_model['depth'], n_classes=self.config_file['n_classes'])
        param = torch.load(self.config_model['dir'])
        delList= self.config_model['delList']
        mlp_param = OrderedDict()
        new_param = ["mlp_head.0.weight", "mlp_head.0.bias", "mlp_head.1.weight", "mlp_head.1.bias"]

        for i in delList:
            mlp_param[new_param[delList.index(i)]] = param[i]
            param.pop(i, None)
            
        model.load_state_dict(param)
        mlp = self.mlp.load_state_dict(mlp_param)

        return model, mlp 

    def test(self, model, skeleton_data):
        model = model.to(self.device)
        model.eval()
        conf_mat = torch.zeros(self.config_file['n_classes'], self.config_file['n_classes'])
        corr = 0
        running_loss = 0
        accuracy = 0
        index = 0

        batch_size = 32

        for b_index in range(int(len(skeleton_data.heat_npy) // batch_size)+1):

            skeleton_data.load_batch(b_index=b_index*batch_size, b_size=batch_size)
            skeleton_dataloader = DataLoader(skeleton_data, batch_size=batch_size, num_workers=2, shuffle=True)
            heat_data, labels = next(iter(skeleton_dataloader))

            inputs = heat_data.type(torch.cuda.FloatTensor).to(self.device)
            label = labels.type(torch.LongTensor).to(self.device)

            with torch.no_grad():
                before_mlp = model(inputs)
                prediction = self.mlp(before_mlp)
                loss = self.criterion(prediction, label)

            running_loss += loss.item()
            prediction = prediction.argmax(1)

            for p, l in zip(prediction.to('cpu'), label.to('cpu')):
                conf_mat[l, p] += 1
                if p == l:
                    corr += 1
                index += 1
            
        
        accuracy = corr / index * 100
        running_loss /= len(skeleton_data)
        return running_loss, accuracy, conf_mat 
    
    def test_top5(self, model, skeleton_data):
        model = model.to(self.device)
        model.eval()
        corr = 0
        running_loss = 0
        accuracy = 0
        index = 0

        batch_size = 32

        for b_index in range(int(len(skeleton_data.heat_npy) // batch_size)):
            
            skeleton_data.load_batch(b_index=b_index*batch_size, b_size=batch_size)
            skeleton_dataloader = DataLoader(skeleton_data, batch_size=batch_size, num_workers=2, shuffle=True)
            heat_data, labels = next(iter(skeleton_dataloader))

            inputs = heat_data.type(torch.cuda.FloatTensor).to(self.device)
            label = labels.type(torch.LongTensor).to(self.device)

            with torch.no_grad():
                before_mlp = model(inputs)
                prediction = self.mlp(before_mlp)
                loss = self.criterion(prediction, label)

            running_loss += loss.item()
            top5, indices = torch.topk(prediction, 5)
            print(indices, label)
            for t, l in zip(indices.to('cpu'), label.to('cpu')):
                if l in t:
                    corr += 1
                index += 1           
        
        accuracy = corr / index * 100
        running_loss /= len(skeleton_data)
        return running_loss, accuracy
            


### imu ### 

class ImuWrapper():
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.config_file['input']['imu']
        self.config_model = self.config_file['model']['imu']
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mlp = mlp_head(self.config_model['dim'], self.config_file['n_classes']).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def load_Dataset(self, train=False):
        if train:
            tdatalist = pd.read_csv(self.config_data['train_dir'], index_col=None)
        else:
            tdatalist = pd.read_csv(self.config_data['label_dir'], index_col=None)
        dts = [d for d in list(tdatalist['imu_path'])]
        labs = list(tdatalist['label'])

        sw_data = many_sliding_window(dts, labs, self.config_data['sw_depth'])
        
        dataset = imuDataset(sw_data)
        # dataLoader = DataLoader(dataset, batch_size=self.config_file['batch_size'], shuffle=train)
        dataLoader = DataLoader(dataset, batch_size=self.config_file['batch_size'], shuffle=False) # 테스트용

        return dataset, dataLoader
    
    def load_Dataset_noyaml(self, dpath): # 나중에 지우기
        tdatalist = pd.read_csv(dpath, index_col=None)
        dts = [d for d in list(tdatalist['imu_path'])]
        labs = list(tdatalist['label'])

        sw_data = many_sliding_window(dts, labs, self.config_data['sw_depth'])
        dataset = imuDataset(sw_data)
        dataLoader = DataLoader(dataset, batch_size=self.config_file['batch_size'], shuffle=False)
        
        return dataLoader 


        
    def load_Model(self, window_size, num_channels, data_depth):
        model = ViTmerge(
            image_size = (window_size, num_channels),
            patch_size = num_channels,
            num_classes = self.config_file['n_classes'],
            channels = data_depth, 
            dim = self.config_model['dim'],
            depth = self.config_model['depth'],
            heads = self.config_model['heads'],
            mlp_dim = self.config_model['mlp_dim'],
            dropout = self.config_model['dropout'],
            emb_dropout = self.config_model['emb_dropout'],
            pool = 'mean'
        )

        param = torch.load(self.config_model['dir'])
        delList= self.config_model['delList']

        for i in delList:
            param.pop(i, None)
        model.load_state_dict(param)
    
        return model

    def load_Model_mlp(self, window_size, num_channels, data_depth):
        model = ViTmerge(
            image_size = (window_size, num_channels),
            patch_size = num_channels,
            num_classes = self.config_file['n_classes'],
            channels = data_depth,
            dim = self.config_model['dim'],
            depth = self.config_model['depth'],
            heads = self.config_model['heads'],
            mlp_dim = self.config_model['mlp_dim'],
            dropout = self.config_model['dropout'],
            emb_dropout = self.config_model['emb_dropout'],
            pool = 'mean'
        )

        param = torch.load(self.config_model['dir'])
        delList = self.config_model['delList']
        mlp_param = OrderedDict()

        for i in delList:
            mlp_param[i] = param[i]
            param.pop(i, None)
            
        model.load_state_dict(param)
        mlp = self.mlp.load_state_dict(mlp_param)

        return model, mlp
        
    def test(self, model, dataloader):
        model = model.to(self.device)
        model.eval()
        test_loss, corr = 0, 0
        accuracy = 0
        index = 0
        conf_mat = torch.zeros(self.config_file['n_classes'], self.config_file['n_classes'])

        for data, target in dataloader:
            data, target = data.permute(0, 3, 1, 2).to(self.device), target.to(self.device)
            
            with torch.no_grad():
                before_mlp = model(data)
                pred = self.mlp(before_mlp)
                loss = self.criterion(pred, target)

            test_loss += loss.item()
            pred = pred.argmax(1)

            for p, l in zip(pred.to('cpu'), target.to('cpu')):
                conf_mat[l, p] += 1
                if p == l:
                    corr += 1
                index += 1
            
            test_loss /= len(dataloader)
            accuracy = corr / index * 100

        return test_loss, accuracy, conf_mat 


def chg_label(orig_labels):

    # original action labels
    # {0: 'Liftcuphandle', 1:'Hairbrush', 2:'Brushteeth', 3:'Remotecon', 4:'MovingCan',
    #  5:'Writing', 6:'FoldingPaper', 7:'Folduptowel', 8:'Washface', 9:'Smartphone', 10:'RightShoulderSide', 
    #  11:'LeftShoulderSide', 12:'RightShoulderFrontal', 13:'LeftShoulderFrontal', 14:'LateralRotation'}

    # merged: 1-2 brush, 6-7 folding
    # {0: 'Liftcuphandle', 1:'Brush', 2:'Remotecon', 3:'MovingCan',
    #  4:'Writing', 5:'Folding', 6:'Washface'}

    change_labels = {0:0, 1:1, 2:1, 3:2, 4:3, 5:4, 6:5, 7:5, 8:6}
    
    for i in range(len(orig_labels)):
        orig_labels[i] = change_labels[int(orig_labels[i])]

    return orig_labels 


### print ###

class ResultWrapper():
    def __init__(self, config_file):
        self.config_file = config_file
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.CrossEntropyLoss()
        self.mlp = mlp_head(self.config_file['model']['imu']['dim'], self.config_file['n_classes'])
        self.lr = self.config_file['learning_rate']
        self.batch_size = self.config_file['batch_size']


    def _get_inference(self, imu_model, imu_dataloader, skeleton_model, skeleton_data, mlp=None):
        imu_model = imu_model.to(self.device)
        skeleton_model = skeleton_model.to(self.device)

        imu_model.eval()
        skeleton_model.eval()

        if mlp == None:
            mlp = self.mlp
            mlp = torch.load(self.config_file['model']['mlp']['dir'])

        mlp = mlp.to(self.device)
        mlp.eval()

        total_loss = 0
        
        all_preds = []
        all_labs =[]
        conf_mat = torch.zeros(self.config_file['n_classes'], self.config_file['n_classes'])
        corr = 0

        for b_index, imu_pack in enumerate(imu_dataloader):
            data, target = imu_pack
            batch_size = data.shape[0]
            
            skeleton_data.load_batch(b_index=b_index*self.batch_size, b_size=batch_size)
            skeleton_dataloader = DataLoader(skeleton_data, batch_size=batch_size, num_workers=2, shuffle=False)
            heat_data, labels = next(iter(skeleton_dataloader))

            # target, labels = chg_label(target), chg_label(labels)
            
            data, target = data.permute(0, 3, 1, 2).to(self.device), target.to(self.device)
            inputs = heat_data.type(torch.cuda.FloatTensor).to(self.device)
            label = labels.type(torch.LongTensor).to(self.device)

            if torch.equal(target, label):
                print('imu-skeleton same label')
            else:
                print('imu', target)
                print('ske', labels)
                break

            with torch.no_grad():
                imu_before_mlp = imu_model(data)
                skeleton_before_mlp = skeleton_model(inputs)

                with open('./chk-mlp.log', 'a') as f:
                    f.write('IMU\n')
                    f.write(str(imu_before_mlp.shape))
                    f.write('\n')
                    f.write(str(imu_before_mlp))
                    f.write('Skeleton\n')
                    f.write(str(skeleton_before_mlp.shape))
                    f.write('\n')
                    f.write(str(skeleton_before_mlp))

                
                hapInfer = torch.cat((imu_before_mlp, skeleton_before_mlp), 1)
                hapInfer = hapInfer / 2 

                print(hapInfer.shape)
                
                pred = mlp(hapInfer)

                loss = self.criterion(pred, target)

                total_loss += loss.item()
                pred = torch.argmax(pred, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labs.extend(label.cpu().numpy())

                for p, l in zip(pred.to('cpu'), target.to('cpu')):
                    conf_mat[l, p] += 1
                    if p == l:
                        corr += 1
                    
                # print(pred) # for check
                # print("imu labels", target)
                # print("skeleton labels", label)

        acc = accuracy_score(all_labs, all_preds)
        cf_matrix = confusion_matrix(all_labs, all_preds)
        print('confusion matrix \n', cf_matrix, '\n self-made confusion matrix\n', conf_mat)

        print(f'loss: {total_loss/len(imu_dataloader)}, accuracy: {acc}')

        # logger
        test_logger = logging.getLogger(name="TEST_LOG")
        test_logger.setLevel(logging.INFO)

        formatter = logging.Formatter('|%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        test_file_handler = logging.FileHandler('./modeltest-results/test_log.log')
        test_file_handler.setFormatter(formatter)

        test_logger.addHandler(test_file_handler)
        test_logger.info(f'test loss: {total_loss/len(imu_dataloader)}, test_accuracy: {acc}')
        test_logger.info(f'test confusion matrix: \n{cf_matrix}')

        return total_loss/len(imu_dataloader), acc, conf_mat


    def get_inference(self, imu_model, imu_dataloader, skeleton_model, skeleton_data, mlp=None):
        imu_model = imu_model.to(self.device)
        skeleton_model = skeleton_model.to(self.device)

        imu_model.eval()
        skeleton_model.eval()

        if mlp == None:
            mlp = self.mlp
            mlp = torch.load(self.config_file['model']['mlp']['dir'])

        mlp = mlp.to(self.device)
        mlp.eval()

        total_loss = 0
        
        all_preds = []
        all_labs =[]
        conf_mat = torch.zeros(self.config_file['n_classes'], self.config_file['n_classes'])
        corr = 0

        for b_index, imu_pack in enumerate(imu_dataloader):
            data, target = imu_pack
            batch_size = data.shape[0]
            
            skeleton_data.load_batch(b_index=b_index*self.batch_size, b_size=batch_size)
            skeleton_dataloader = DataLoader(skeleton_data, batch_size=batch_size, num_workers=2, shuffle=False)
            heat_data, labels = next(iter(skeleton_dataloader))

            # target, labels = chg_label(target), chg_label(labels)
            
            data, target = data.permute(0, 3, 1, 2).to(self.device), target.to(self.device)
            inputs = heat_data.type(torch.cuda.FloatTensor).to(self.device)
            label = labels.type(torch.LongTensor).to(self.device)

            if torch.equal(target, label):
                print('imu-skeleton same label')
            else:
                print('imu', target)
                print('ske', labels)
                break

            with torch.no_grad():
                imu_before_mlp = imu_model(data)
                skeleton_before_mlp = skeleton_model(inputs)
                
                imu_linear = nn.Linear(31, 64).to(self.device)
                ske_linear = nn.Linear(257, 64).to(self.device)
                
                imu_before_mlp = (imu_before_mlp - imu_before_mlp.mean()) / imu_before_mlp.std()
                imu_before_mlp = imu_before_mlp.transpose(1, 2)
                imu_before_mlp = imu_linear(imu_before_mlp)
                imu_before_mlp = imu_before_mlp.transpose(1, 2)

                skeleton_before_mlp = (skeleton_before_mlp - skeleton_before_mlp.mean()) / skeleton_before_mlp.std()
                skeleton_before_mlp = skeleton_before_mlp.transpose(1, 2)
                skeleton_before_mlp = ske_linear(skeleton_before_mlp)
                skeleton_before_mlp = skeleton_before_mlp.transpose(1, 2)

                hapInfer = torch.cat((imu_before_mlp, skeleton_before_mlp), 1)
                hapInfer = hapInfer / 2 
                
                pred = mlp(hapInfer)

                loss = self.criterion(pred, target)

                total_loss += loss.item()
                pred = torch.argmax(pred, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labs.extend(label.cpu().numpy())

                for p, l in zip(pred.to('cpu'), target.to('cpu')):
                    conf_mat[l, p] += 1
                    if p == l:
                        corr += 1
                    
        acc = accuracy_score(all_labs, all_preds)
        cf_matrix = confusion_matrix(all_labs, all_preds)
        print('confusion matrix \n', cf_matrix, '\n self-made confusion matrix\n', conf_mat)

        print(f'loss: {total_loss/len(imu_dataloader)}, accuracy: {acc}')

        # logger
        test_logger = logging.getLogger(name="TEST_LOG")
        test_logger.setLevel(logging.INFO)

        formatter = logging.Formatter('|%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        test_file_handler = logging.FileHandler('./modeltest-results/test_log.log')
        test_file_handler.setFormatter(formatter)

        test_logger.addHandler(test_file_handler)
        test_logger.info(f'test loss: {total_loss/len(imu_dataloader)}, test_accuracy: {acc}')
        test_logger.info(f'test confusion matrix: \n{cf_matrix}')

        return total_loss/len(imu_dataloader), acc, cf_matrix

    def mlp_train(self, imu_model, imu_dataloader, skeleton_model, skeleton_data, mlp):
        imu_model = imu_model.to(self.device)
        skeleton_model = skeleton_model.to(self.device)

        # parameter freeze parameter of imu/skeleton model
        for param in imu_model.parameters():
            param.requires_grad = False
        for param in skeleton_model.parameters():
            param.requires_grad = False

        mlp = mlp.to(self.device)
        mlp.train()

        optimizer = torch.optim.Adam(mlp.parameters(), lr=float(self.lr))

        total_loss = 0.0
        all_preds = []
        all_labs =[]

        batch_idx = 0

        for b_index, imu_pack in enumerate(imu_dataloader):
            data, target = imu_pack
            batch_size = data.shape[0]
            
            skeleton_data.load_batch(b_index=b_index*batch_size, b_size=batch_size)
            skeleton_dataloader = DataLoader(skeleton_data, batch_size=batch_size, num_workers=2, shuffle=True)
            heat_data, labels = next(iter(skeleton_dataloader))

            # target, labels = chg_label(target), chg_label(labels)

            data, target = data.permute(0,3,1,2).to(self.device), target.to(self.device)
            inputs = heat_data.type(torch.cuda.FloatTensor).to(self.device)
            label = labels.type(torch.LongTensor).to(self.device)

            with torch.no_grad():
                imu_before_mlp = imu_model(data)
                skeleton_before_mlp = skeleton_model(inputs)

                imu_linear = nn.Linear(31, 64).to(device=self.device)
                ske_linear = nn.Linear(257, 64).to(device=self.device)
                
                imu_before_mlp = (imu_before_mlp - imu_before_mlp.mean()) / imu_before_mlp.std()
                imu_before_mlp = imu_before_mlp.transpose(1, 2)
                imu_before_mlp = imu_linear(imu_before_mlp)
                imu_before_mlp = imu_before_mlp.transpose(1, 2)

                skeleton_before_mlp = (skeleton_before_mlp - skeleton_before_mlp.mean()) / skeleton_before_mlp.std()
                skeleton_before_mlp = skeleton_before_mlp.transpose(1, 2)
                skeleton_before_mlp = ske_linear(skeleton_before_mlp)
                skeleton_before_mlp = skeleton_before_mlp.transpose(1, 2)

            # imu:skeleton=1:1로 섞기
            hapInfer = torch.cat((imu_before_mlp, skeleton_before_mlp), 1)
            hapInfer = hapInfer / 2

            pred = mlp(hapInfer)
            loss = self.criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(pred, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labs.extend(label.cpu().numpy())

            batch_idx += 1

        acc = accuracy_score(all_labs, all_preds)
        print(f'loss: {total_loss/len(imu_dataloader)}, accuracy: {acc}')
        
        return acc

    def mlp_train_and_save(self, imu_model, imu_dataloader, imu_testloader, skeleton_model, skeleton_data, skeleton_test, mlp_head):
        train_epoch = self.config_file['train_epochs']

        best_loss = 1000
        mlp_to_train = mlp_head
        
        train_loss_list = []
        test_loss_list = []
        test_accu_list = []

        # logger
        train_logger = logging.getLogger(name="TRAIN_LOG")
        train_logger.setLevel(logging.INFO)

        formatter = logging.Formatter('|%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        train_file_handler = logging.FileHandler('./modeltest-results/train_log.log')
        train_file_handler.setFormatter(formatter)

        train_logger.addHandler(train_file_handler)


        for epoch in range(train_epoch):
            print(f"===EPOCH {epoch+1}/{train_epoch}===")
            train_loss = self.mlp_train(imu_model, imu_dataloader, skeleton_model, skeleton_data, mlp_to_train)
            
            test_loss, test_accuracy, test_cm = self.get_inference(imu_model, imu_testloader, skeleton_model, skeleton_test, mlp_to_train)

            train_logger.info(f'EPOCH {epoch+1}/{train_epoch}')
            train_logger.info(f'train loss: {train_loss}, valid_loss: {test_loss}, valid_accuracy: {test_accuracy}')
            train_logger.info(f'valid confusion matrix: {test_cm}')
            
            torch.save(mlp_to_train, './model-save/mlp_latest.pth')
            if test_loss < best_loss:
                best_loss = test_loss 
                torch.save(mlp_to_train, './model-save/mlp_best.pth')
                torch.save(mlp_to_train, './modeltest-results/mlp_best.pth')
                torch.save(mlp_to_train.state_dict(), './modeltest-results/mlp_best_dict.pth')

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            test_accu_list.append(test_accuracy)

        fig, ax = plt.subplots(1,2, figsize=(14, 7))
        epoch_li = [i for i in range(train_epoch)]

        ax[0].plot(train_loss_list, 'o-', label="train loss")
        ax[0].plot(test_loss_list, 'o-', label="test loss")
        ax[0].legend()
        ax[0].set_ylim(0, 0.5)
        ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("loss")
        ax[0].set_title("Test Loss")

        ax[1].plot(test_accu_list, 'o-', color="orange")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].set_title("Model Accuracy")
        plt.savefig('./modeltest-results/result-plot.png')
        plt.close()


    def show_confusion_matrix(self, cf_matrix):
        
        action_labels = {0: 'Liftcuphandle', 1:'Hairbrush', 2:'Brushteeth', 3:'Remotecon', 4:'MovingCan',
                5:'Writing', 6:'FoldingPaper', 7:'Folduptowel', 8:'Washface'}

        # merged labels
        # action_labels = {0: 'Liftcuphandle', 1:'Brush', 2:'Remotecon', 3:'MovingCan',
        #        4:'Writing', 5:'Folding', 6:'Washface'}

        cf_matrix = pd.DataFrame(cf_matrix, index=list(action_labels.values()), columns=list(action_labels.values()))
        sns.heatmap(cf_matrix, cmap='Blues', annot=True, annot_kws={"size":8}, fmt='d')
        plt.xticks(rotation=90)
        plt.title("Confusion matrix")
        plt.ylabel("True")
        plt.xlabel("Prediction")
        plt.savefig("./modeltest-results/confusion_matrix.png", bbox_inches="tight")
