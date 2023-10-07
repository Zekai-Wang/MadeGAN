import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        with torch.no_grad():
            nn.init.xavier_normal_(mod.weight)

    elif classname.find('BatchNorm') != -1:
        with torch.no_grad():
            mod.weight.normal_(1.0, 0.02)
            mod.bias.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        with torch.no_grad():
            mod.bias.fill_(0.01)
              
class Encoder(nn.Module):
    def __init__(self, ngpu, out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(1,16,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 16 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(16 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16 * 2, 16 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(16 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16 * 4, 16 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(16 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16 * 8, 32 * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(32 * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32 * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
        )
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

class Decoder(nn.Module):
    def __init__(self, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(50,32*16,10,1,0,bias=False),
            nn.BatchNorm1d(32*16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(32 * 16, 16 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(16 * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(16 * 8, 16 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(16 * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(16 * 4, 16*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(16*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(16 * 2, 16 , 4, 2, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
    
class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )
        
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        x = x.view(-1, s[1])
        #
        y_and = self.memory(x)
        #
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')
        return {'output': y, 'att': att}

# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxDxHxW --> NxDxHxW x C
    if l == 2:
        x = input # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxDxHxW x C --> (NxDxHxW) x C
    x = x.view(-1, s[1])
    return x

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val
    
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        model = Encoder(1,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features
        
class Generator(nn.Module):

    def __init__(self, shrink_thres=0.0025,mem_dim=2000):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(1,50)
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=50, shrink_thres =shrink_thres)
        self.decoder = Decoder(1)

    def forward(self, x):
        latent_0 = self.encoder1(x)
        latent_dict = self.mem_rep(latent_0)
        att = latent_dict['att']
        latent_i = latent_dict['output']
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i, att, latent_0
