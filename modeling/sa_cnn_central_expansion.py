import torch
import torch.nn as nn
import numpy as np

class self_attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,q_dim,k_dim,v_dim,v_mapping,activation='relu'):
        super(self_attention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.v_mapping = v_mapping

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = q_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = k_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = v_dim , kernel_size= 1)
        self.o_proj = nn.Conv2d(in_channels = v_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)        
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        if self.v_mapping == True:
            proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        else:
            proj_value = x.view(m_batchsize,-1,width*height) # B X C X N


        out = torch.bmm(proj_value,attention.permute(0,2,1))
        
        if self.v_mapping == True:
            out = out.view(m_batchsize,self.v_dim,width,height)
            out = self.o_proj(out)
        else:
            out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class net_one_neuron_sa_central_expansion(nn.Module):
    def __init__(self,q_dim,k_dim,v_dim,v_mapping,expand_dim):
        super().__init__()
        self.layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(5, 5), stride=(1, 1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3)
         ) #[N,30,9,9] 
        
        self.attention = self_attention(in_dim=30,q_dim=q_dim,k_dim=k_dim,v_dim=v_dim,v_mapping=v_mapping)

        self.layers_sa_reg = nn.Sequential(
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3)
        )

        self.layers_2 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3), #or here
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
        )
        self.Linear1 = nn.Linear(30, expand_dim)
        self.Linear2 = nn.Linear(expand_dim,1)

    def forward(self, x):
        x = self.layers_1(x)
        x,_ = self.attention(x) 
        x = self.layers_sa_reg(x)
        x = self.layers_2(x)
        x = x.reshape(-1,30,25)
        x = x[:,:,12]
        x = self.Linear1(x)
        x = self.Linear2(x)
        return x


class seperate_core_model_sa_central_expansion(nn.Module):
    def __init__(self,num_neurons,q_dim,k_dim,v_dim,v_mapping,expand_dim):
        super().__init__()
        self.models = nn.ModuleList([net_one_neuron_sa_central_expansion(q_dim=q_dim,k_dim=k_dim,v_dim=v_dim,v_mapping=v_mapping,expand_dim=expand_dim) for i in range(num_neurons)])
        self.num_neurons = num_neurons

    def forward(self, x):
        outputs = [self.models[i].forward(x) for i in range(self.num_neurons)]
        outputs = torch.stack(outputs, dim=1)
        return outputs.reshape((outputs.shape[0], outputs.shape[1]))
