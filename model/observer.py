from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

from Utils.conv1x1 import one_one_conv


# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys
from torch.nn.parameter import Parameter
from helpers import *

print('Observer: initialized.')



class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
       
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 
class obs_Decoder(nn.Module):
      def __init__(self, mdim):
        super(obs_Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.fc = nn.Conv2d
      def forward(self, r4, r3, r2,feat):
        #feat = [torch.zeros_like(f) for f in feat]   --> 디버깅해보고 넣을지 말지 
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4+feat[0]) # out: 1/8, 256    ////////////////////////////// plus feat[0] === m4 
        m2 = self.RF2(r2, m3+feat[1]) # out: 1/4, 256     ////////////////////////////// plus feat[1] === m3 

        p2 = self.pred2(F.relu(m2+feat[2])) #              ////////////////////////////// plus feat[2] === m2 

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        '''
        p=p.detach().cpu()
        unc_fc = self.fc(num_objects,num_objects,kernel_size=1,stride=1,padding=0)
        uncertainty = unc_fc(p)
        print("uncertainty=obs_ps's shape: {}".format(uncertainty.shape))
        '''
        return p #, p2, p3, p4



class Observer(nn.Module):
    
    def __init__(self):
        super(Observer, self).__init__()
        self.obs_Decoder = obs_Decoder(256) #added for obs
        
    def one_one_conv(self,obs_logit,num_objects):
        fc = nn.Conv2d(num_objects, 1, kernel_size=1, stride=1, padding=0)
        #filter=torch.ones(1,1,1)
        #fc.weight = Parameter(filter)
        fc.weight.data.fill_(1)
        print("one by one conv weight print!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(fc.weight.data)
        return fc(obs_logit)
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems


    def Soft_aggregation(self, ps, K):
        num_objects, H, W = ps.shape
        em = ToCuda(torch.zeros(1, K, H, W)) 
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        em[0,1:num_objects+1] = ps # obj prob
        em = torch.clamp(em, 1e-7, 1-1e-7)
        logit = torch.log((em /(1-em)))
        return logit
   
    def segment(self, frame, keys, values, num_objects,feats,m4,r3e,r2e): 
        ### layers freeze start 
        #self.layer_freeze()


        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B = 1
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        obs_logits = self.obs_Decoder(m4, r3e, r2e,feats) 


        #obs_ps = F.sigmoid(obs_logits)        
        obs_ps = F.softmax(obs_logits, dim=1)[:,1] # no, h, w  
        obs_logit = self.Soft_aggregation(obs_ps,K)
        
        if pad[2]+pad[3] > 0:
            obs_logit = obs_logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            obs_logit = obs_logit[:,:,:,pad[0]:-pad[1]]

        observer_output = one_one_conv(obs_logit,num_objects)

        #STM frozen output : logit  / obs output(uncertainty) : obs_ps


        return observer_output

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        

