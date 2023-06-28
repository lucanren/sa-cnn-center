import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from prettytable import PrettyTable


def pc_one_neuron(net,neuron_num,vimg,vresp):
    real_resp = np.asarray([item[neuron_num:neuron_num+1] for item in vresp])
    images = torch.from_numpy(np.float32(vimg))
    with torch.no_grad():
        net.eval()
        net_resp = net(images).detach().numpy()
    real_resp = np.transpose(real_resp)
    net_resp = np.transpose(net_resp)
    corrs = []
    corrs.append(np.corrcoef(real_resp[0], net_resp[0])[0][1])
    return np.mean(corrs)

def pc_firstx_neuron(net,firstx,vimg,vresp):
    real_resp = np.asarray([item[0:firstx] for item in vresp])
    images = torch.from_numpy(np.float32(vimg))
    with torch.no_grad():
        net.eval()
        net_resp = net(images).detach().numpy()
    real_resp = np.transpose(real_resp)
    net_resp = np.transpose(net_resp)
    corrs = []
    for i in range(firstx):
        corrs.append(np.corrcoef(real_resp[i], net_resp[i])[0][1])
    return np.mean(corrs),corrs

class ImageDataset(Dataset):
    def __init__(self, data, labels, num_neurons):
        self.data = data
        self.labels = labels
        self.num_neurons = num_neurons

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = torch.tensor(self.data[index], dtype=torch.float)
        label = torch.tensor(self.labels[index, 0:self.num_neurons], dtype=torch.float)
        return img, label
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
