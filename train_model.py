import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader
from modeling.sa_cnn_central import *
from modeling.sa_cnn_central_expansion import *
from modeling.sa_cnn_central_halfumap import *
from modeling.sa_cnn_central_halfumap_2 import *
from utils import *
from GS_functions import GF

num_neurons = 10
train_img = np.load('../all_sites_data_prepared/pics_data/train_img_m1s1.npy')
train_resp = np.load('../all_sites_data_prepared/New_response_data/trainRsp_m1s1.npy')
train_img=np.reshape(train_img,(34000,1,50,50))
train_dataset = ImageDataset(train_img,train_resp,num_neurons)
train_loader = DataLoader(train_dataset,50,shuffle=True)

val_img = np.load('../all_sites_data_prepared/pics_data/val_img_m1s1.npy')
val_resp = np.load('../all_sites_data_prepared/New_response_data/valRsp_m1s1.npy')
val_img = np.reshape(val_img,(1000,1,50,50))
val_dataset = ImageDataset(val_img,val_resp,num_neurons)
val_loader = DataLoader(val_dataset,50,shuffle=True)

MAXCORRSOFAR = 0

Q,K,V,V_MAPPING = 30,30,30,True

# savepathname=f'sa_cnn_central_q_k_v_vm_exp_{Q}_{K}_{V}_{V_MAPPING}'
savepathname=f'sa_cnn_central_q_k_v_front_half_mapped_{Q}_{K}_{V}'

GF.mkdir('./results/',savepathname)
log = open(f'./results/{savepathname}/log.txt', "w")

net = seperate_core_model_sa_central_halfumap_2(num_neurons=num_neurons,q_dim=Q,k_dim=K,v_dim=V,v_mapping=V_MAPPING)
opt = torch.optim.Adam(net.parameters(),lr=0.001)
lfunc = torch.nn.MSELoss()

losses = []
accs = []
for epoch in range(50): 
    net.train()
    train_losses=[]
    for x,y in train_loader:
        l = lfunc(net(x),y)
        opt.zero_grad()
        l.backward()
        opt.step()
        train_losses.append(l.item())
    losses.append(np.mean(train_losses))
    
    val_losses=[]
    with torch.no_grad():
        net.eval()
        for x,y in val_loader:
            l = lfunc(net(x),y)
            val_losses.append(l.item())
    accs.append(np.mean(val_losses))

    log.write("Epoch " + str(epoch) + " train loss is:" + str(losses[-1])+"\n")
    log.write("Epoch " + str(epoch) + " val loss is:" + str(accs[-1])+"\n")
    print("Epoch " + str(epoch) + " train loss is:" + str(losses[-1])+"\n")
    print("Epoch " + str(epoch) + " val loss is:" + str(accs[-1])+"\n")

    mean_corrs,corrs =  pc_firstx_neuron(net,num_neurons,val_img,val_resp)

    log.write("Epoch " + str(epoch) + " mean correlation is:" + str(mean_corrs)+"\n")
    log.write("Epoch " + str(epoch) + " correlations:" + str(corrs)+"\n")
    print("Epoch " + str(epoch) + " mean correlation is:" + str(mean_corrs)+"\n")
    print("Epoch " + str(epoch) + " correlations:" + str(corrs)+"\n")
    
    if mean_corrs > MAXCORRSOFAR:
        MAXCORRSOFAR=np.mean(corrs)
        log.write("Epoch " + str(epoch) + ": saved sucessfully.\n")
        print("Epoch " + str(epoch) + ": saved sucessfully.\n")
        torch.save(net.state_dict(),  f'./results/{savepathname}/model.pth')
    log.write("\n")
    print("\n")

log.close()