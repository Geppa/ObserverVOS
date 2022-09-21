from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
def one_one_conv(obs_logit,num_objects):
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    fc = nn.Conv2d(11, 1, kernel_size=1, stride=1, padding=0)
    fc=fc.to(device)
    obs_logit=obs_logit.to(device)
    #obs_logit=torch.tensor(obs_logit)
    #one = torch.Tensor(1).to(device)
    #fc.weight.data.fill_(1)

    #print("one by one conv weight print!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(fc.weight.data)
    return fc(obs_logit)