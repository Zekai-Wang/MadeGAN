from Module import *
from functions import *

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold

import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from  torch.utils.data import DataLoader,TensorDataset

from sklearn.model_selection import train_test_split
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


#Data Set
N_samples = np.load( "N_samples.npy") #NxCxL
S_samples = np.load( "S_samples.npy")
V_samples = np.load("V_samples.npy")
F_samples = np.load("F_samples.npy")

# normalize all
for i in range(N_samples.shape[0]):
    for j in range(1):
        N_samples[i][j]=normalize(N_samples[i][j][:])
N_samples=N_samples[:,:1,:]

for i in range(S_samples.shape[0]):
    for j in range(1):
        S_samples[i][j] = normalize(S_samples[i][j][:])
S_samples = S_samples[:, :1, :]

for i in range(V_samples.shape[0]):
    for j in range(1):
        V_samples[i][j] = normalize(V_samples[i][j][:])
V_samples = V_samples[:, :1, :]

for i in range(F_samples.shape[0]):
    for j in range(1):
        F_samples[i][j] = normalize(F_samples[i][j][:])
F_samples = F_samples[:, :1, :]


# train / test
val_S, val_S_y = S_samples, np.ones((S_samples.shape[0], 1))
val_V, val_V_y = V_samples, np.ones((V_samples.shape[0], 1))
val_F, val_F_y = F_samples, np.ones((F_samples.shape[0], 1))


# train / val
N_y = np.zeros((N_samples.shape[0], 1))
train_N, val_N, train_N_y, val_N_y = getPercent(N_samples, N_y, 0.17, 0)

val_data=np.concatenate([val_N,val_S,val_V,val_F])
val_y=np.concatenate([val_N_y,val_S_y,val_V_y,val_F_y])

train_dataset = TensorDataset(torch.Tensor(train_N),torch.Tensor(train_N_y))
val_dataset= TensorDataset(torch.Tensor(val_data), torch.Tensor(val_y))
test_N_dataset = TensorDataset(torch.Tensor(val_N), torch.Tensor(val_N_y))
test_S_dataset = TensorDataset(torch.Tensor(val_S), torch.Tensor(val_S_y))
test_V_dataset = TensorDataset(torch.Tensor(val_V), torch.Tensor(val_V_y))
test_F_dataset = TensorDataset(torch.Tensor(val_F), torch.Tensor(val_F_y))

print("train data size:{}".format(train_N.shape))
print("val data size:{}".format(val_data.shape))
print("val N data size:{}".format(val_N.shape))
print("val S data size:{}".format(val_S.shape))
print("val V data size:{}".format(val_V.shape))
print("val F data size:{}".format(val_F.shape))


batchsize = 64
dataloader = {"train": DataLoader(
                        dataset=train_dataset,  # torch TensorDataset format
                        batch_size=batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(1),
                        drop_last=True),
                    "val": DataLoader(
                        dataset=val_dataset,  # torch TensorDataset format
                        batch_size=batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(1),
                        drop_last=False),
                    "test_N":DataLoader(
                            dataset=test_N_dataset,  # torch TensorDataset format
                            batch_size=batchsize,  # mini batch size
                            shuffle=True,
                            num_workers=int(1),
                            drop_last=False),
                    "test_S": DataLoader(
                        dataset=test_S_dataset,  # torch TensorDataset format
                        batch_size=batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(1),
                        drop_last=False),
                    "test_V": DataLoader(
                        dataset=test_V_dataset,  # torch TensorDataset format
                        batch_size=batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(1),
                        drop_last=False),
                    "test_F": DataLoader(
                        dataset=test_F_dataset,  # torch TensorDataset format
                        batch_size=batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(1),
                        drop_last=False),
                    }

train_hist1 = {}
train_hist1['D_loss'] = []
train_hist1['G_loss'] = []
train_hist1['val_auc'] = []
train_hist1['val_prc'] = []
train_hist1['per_epoch_time'] = []
train_hist1['total_time'] = []

device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")

start_time = time.time()
best_auc=0
best_auc_epoch=0
total_steps = 0
cur_epoch=0

#Generator
netG = Generator().to(device)
netG.apply(weights_init)

#Discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)

bce_criterion = nn.BCELoss()
mse_criterion=nn.MSELoss()
#tr_entropy_loss_func = EntropyLossEncap().to(device)

optimizerD = optim.Adam(netD.parameters(), lr=0.00002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.00002, betas=(0.5, 0.999))

input1 = torch.empty(size=(64, 1, 320), dtype=torch.float32, device=device)
label = torch.empty(size=(64,), dtype=torch.float32, device=device)
gt    = torch.empty(size=(64,), dtype=torch.long, device=device)
fixed_input = torch.empty(size=(64, 1, 320), dtype=torch.float32, device=device)
real_label = 1
fake_label= 0

out_d_real = None
feat_real = None
fake = None
latent_i = None
latent_0 = None
att = None
out_d_fake = None
feat_fake = None
err_d_real = None
err_d_fake = None
err_d = None
out_g = None
err_g_adv = None
err_g_rec = None
err_g = None
entropy_err = None

## Training/Validation Step

niter = 500
for epoch in range(niter):
    cur_epoch+=1
    epoch_iter = 0
    epoch_start_time = time.time()
    for data in dataloader["train"]:
        total_steps += 64
        epoch_iter += 1
        #set input
        with torch.no_grad():
            input1.resize_(data[0].size()).copy_(data[0])
            gt.resize_(data[1].size()).copy_(data[1])
            if total_steps == 64:
                fixed_input.resize_(input1[0].size()).copy_(input1[0])
        
        ###################
        ####update netD####
        ###################
        netD.zero_grad()
        # Train with real
        label.data.resize_(64).fill_(real_label)
        out_d_real, feat_real = netD(input1)
        # Train with fake
        label.data.resize_(64).fill_(fake_label)
        fake, latent_i, att, latent_0 = netG(input1)
        out_d_fake, feat_fake = netD(fake)
        ###
        out_d_real = out_d_real.to(torch.float32)
        out_d_fake = out_d_fake.to(torch.float32)
        err_d_real = bce_criterion(out_d_real, torch.full((64,), 
                                   real_label, dtype=torch.float32, 
                                   device=device))
        err_d_fake = bce_criterion(out_d_fake, 
                                   torch.full((64,), 
                                   fake_label, 
                                   dtype=torch.float32,
                                   device=device))

        err_d = err_d_real + err_d_fake
        err_d.backward()
        optimizerD.step()
        ###################
        ####update netG####
        ###################
        netG.zero_grad()
        label.data.resize_(64).fill_(real_label)
        fake, latent_i, att, latent_0 = netG(input1)
        out_g, feat_fake = netD(fake)
#        entropy_err = tr_entropy_loss_func(att)
        latent_err = mse_criterion(latent_i,latent_0)
        _, feat_real = netD(input1)
        #feature mapping
        err_g_adv = mse_criterion(feat_fake,feat_real)
        #reconstruction error
        err_g_rec = mse_criterion(fake, input1)
        err_g = err_g_rec + err_g_adv
        err_g.backward()
        optimizerG.step()
        # If netD loss is too low, re-initialize netD
        if err_d.item() < 5e-6:
            netD.apply(weights_init)
        errors = {'err_d': err_d.item(),
                  'err_g': err_g.item(),
                  'err_d_real': err_d_real.item(),
                  'err_d_fake': err_d_fake.item(),
                  'err_g_adv': err_g_adv.item(),
                  'err_g_rec': err_g_rec.item()}
        train_hist1['D_loss'].append(errors["err_d"])
        train_hist1['G_loss'].append(errors["err_g"])
        if (epoch_iter  % 200) == 0:
            print("Epoch: [%d] [%4d/%4d] D_loss(R/F): %.6f/%.6f, G_loss: %.6f" %
                ((cur_epoch), (epoch_iter), dataloader["train"].dataset.__len__() //64,
                errors["err_d_real"], errors["err_d_fake"], errors["err_g"]))
    train_hist1['per_epoch_time'].append(time.time() - epoch_start_time)
    
    
    #######################
    ####Validation Step####
    #######################
    with torch.no_grad():
        an_scores = torch.zeros(size=(len(dataloader["val"].dataset),), 
                                dtype=torch.float32, device=device)
        gt_labels = torch.zeros(size=(len(dataloader["val"].dataset),), 
                                dtype=torch.long, device=device)
        latent_i  = torch.zeros(size=(len(dataloader["val"].dataset), 50), 
                                dtype=torch.float32, device=device)
        dis_feat = torch.zeros(size=(len(dataloader["val"].dataset), 32*16*10), 
                               dtype=torch.float32,
                               device=device)
        
        for i, data in enumerate(dataloader["val"], 0):
            with torch.no_grad():
                input1.resize_(data[0].size()).copy_(data[0])
                gt.resize_(data[1].size()).copy_(data[1])
            fake, latent_i, att, latent_0 = netG(input1)
            out_d_real, feat_real = netD(input1)
            out_d_fake, feat_fake = netD(fake)
            error = torch.mean(
                       torch.pow((input1.view(input1.shape[0], -1) -fake.view(fake.shape[0], -1)), 2),
                       dim=1)
            #error_d = torch.mean(
            #            torch.pow((feat_real.view(feat_real.shape[0], -1) -feat_fake.view(feat_fake.shape[0], -1)), 2),
            #            dim=1)
            an_scores[i*64 : i*64 + error.size(0)] = error.reshape(error.size(0))
            gt_labels[i*64 : i*64 + error.size(0)] = gt.reshape(error.size(0))
            #latent_i[i*64 : i*64 + error.size(0), :] = latent_i.reshape(error.size(0), 50)

        an_scores = (an_scores - torch.min(an_scores)) / (torch.max(an_scores) - torch.min(an_scores))
        y_ = gt_labels.cpu().numpy()
        y_pred = an_scores.cpu().numpy()
        min_score,max_score=np.min(y_pred),np.max(y_pred)
        rocprc,rocauc,best_th,best_f1=evaluate(y_,(y_pred-min_score)/(max_score-min_score))
    train_hist1['val_auc'].append(rocauc)
    train_hist1['val_prc'].append(rocprc)
    if rocauc > best_auc:
        best_auc = rocauc
        best_prc = rocprc
        best_y_pred = y_pred
        best_auc_epoch = cur_epoch
        #torch.save(netG, 'MadeGAN_G.pkl')
        #torch.save(netD, 'MadeGAN_D.pkl')
    print("[{}] auc:{:.4f} prc:{:.4f} th:{:.4f} f1:{:.4f} \t best_auc:{:.4f} best_prc:{:.4f} in epoch[{}]\n".format(cur_epoch,rocauc,rocprc, best_th,best_f1,best_auc,best_prc, best_auc_epoch))

train_hist1['total_time'].append(time.time() - start_time)
print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist1['per_epoch_time']),
                                                                        niter,
                                                                        train_hist1['total_time'][0]))

