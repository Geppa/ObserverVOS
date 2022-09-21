
from __future__ import division
import gc
import torch
from torch.autograd import Variable
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
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

#obsnet
import torch
from torchvision.utils import make_grid
from Utils.affichage import draw, plot 
from Utils.utils import transform_output
from Utils.utils import memory_save
from Utils.adv_attacks import select_attack

#model
from Datasets.obstm_dataset import DAVIS_MO_Train
import torch.optim.lr_scheduler as scheduler
from model.stm_model import STM
from Utils.conv1x1 import one_one_conv
from Utils.utils import stm_segment

#for debugging
import os

#for visualizing the flow of gradient.
#from torchviz import make_dot   

""" Train the observer network for one epoch
        epoch        ->  int: current epoch
        obsnet       ->  torch.Module: the observer to train
        segnet       ->  torch.Module: the stm pretrained and freeze
        train_loader ->  torch.DataLoader: the training dataloader
        optimizer    ->  torch.optim: optimizer to train observer
        writer       ->  SummaryWriter: for tensorboard log
        args         ->  Argparse: global parameters
    return:
        avg_loss -> float : average loss on the dataset
"""

def training(epoch, stan_stm,model, Trainloader, optimizer, writer, args):
  
  
#### Print start training the video.
  
    YEAR = args.y
    SET = args.s
    MODEL = 'Observer'
    code_name = '{}_DAVIS_{}_{}'.format(MODEL,YEAR,SET)
    print('Start Training:', code_name)


##### Train

    model.train()
    avg_loss, nb_sample, obsnet_acc, stm_acc = 0, 0, 0., 0.    #Set values for using globally
 
    for i, V in enumerate(Trainloader):
    
    #### Variable preprocessing 
        Fs, Ms, num_objects, info = V
        seq_name = info['name'][0]
        num_frames = info['num_frames'][0].item()
        print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
        args.nclass = num_objects

        #convert to Observer Variables from STM variables. (to images)
        images = torch.transpose(Fs,3,4)                                          #batch rgb w h 
        Ms_2 = Ms[0]                                                              #class(11)x frame x h x w   // make target variable using Mask variable.
        Ms_2 = torch.transpose(Ms_2,3,2)                                          #class(11) x frame x w x h
        Ms_2 = torch.argmax(Ms_2,dim=0)                                           #frame x w x h
        target = Ms_2

        #Setting for observer net variables
        bsize, channel,frame, width, height = images.size()                       #1,3,w,h / bsize means batchsize
        images=images.cpu() 
        target=target.cpu()
        args.size=[height,width]
        transposed_target=torch.transpose(target,1,2).cpu()                       #Trasnpose the target for using in STM.
        shape_b,shape_class,shape_frame,shape_h,shape_w = Ms.shape
        obs_total_logit = torch.zeros(shape_b,shape_frame,shape_h,shape_w)
        whole_mask=torch.zeros(num_frames,bsize,channel,width,height)
        supervision = torch.zeros(bsize,num_frames,height,width)

        #Setting for STM net variable
        Es = torch.zeros_like(Ms)
        Es[:,:,0] = Ms[:,:,0]



    #### Calculate the memory keys and values with stm inference about current video. it needs for adv attack method and model process.
        with torch.no_grad():
            mem_keys, mem_values,stm_seg,stm_Es = memory_save(stan_stm,Fs,Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
        
        #set the loss variable.
        vid_avg_loss=0.0 
        
        #Set for Parameter Update variable 
        update_chunck = 4                            # number of loss update at once.
        if (num_frames-1)%update_chunck == 0:        # state of no remainder.
            loop_manager = 0                         # no add in out loop range.
        else:
            loop_manager = 1                         # add in out loop range for remainder frames. 
        

    #### Training process on.     
        for out_t in tqdm.tqdm(range(0,((num_frames-1)//update_chunck)+loop_manager)):            
            chunck_counter = 0                                                              # variable for indexing the chunck of prediction and labels.
            pred_chunck = torch.zeros((bsize,update_chunck,height,width)).to(args.device)   # variable for preds
            label_chunck = torch.zeros((bsize,update_chunck,height,width)).to(args.device)  # variable for labels
    
            if out_t == ((num_frames-1)//update_chunck)+loop_manager-1:                     # Last frame chunck. 
                inner_max_t = num_frames                                                    # If it is last chunck,  Inner loop range changes.
            else:
                inner_max_t = (update_chunck*(out_t+1))+1                                   # Mid frame chunck. process will go on.

            for t in range((update_chunck*out_t)+1, inner_max_t):                           # Starts at head of each chunck. Ends at tail of each chunck.
                nb_sample += bsize * height * width                                         # number of every pixel. For using at calculating loss.  

            
            #### Adversarial attack 
                if args.adv != "none":                                                      # perform the LAA
                    images[:,:,t], mask = select_attack(images[:,:,t].to(args.device), target[t].to(args.device),stan_stm,V, args,mem_keys,mem_values,t,stm_seg,stm_Es)

                #Setting for using in STM. 
                whole_mask[t-1] = mask.cpu()
                pert_images=images[:,:,t].cpu()
                pert_images=torch.transpose(pert_images,2,3) 
                
                #images[:,:,t] --> 1 3 w h
                #mask -> 1 3 w h 
                #pert_images -> 1 3 h w 

            #### STM segment inference inserting attacked input.  
                with torch.no_grad():
                    _,Es,logit,m4,r3e,r2e,feats = stm_segment(stan_stm,Fs,pert_images,Ms, num_frames, num_objects,mem_keys,mem_values,t, Mem_every=5, Mem_number=None) 

                    #Set output to use in loss calculating and accuracy calculating 
                    pred = np.argmax(Es[0,:,t].detach().cpu().numpy(), axis=0).astype(np.uint8)     # ES[0,:,t]-> class h w ,  pred --> h w 
                    s_pred=torch.Tensor(pred).contiguous()                                          # s_pred --> h w
                    transposed_target=transposed_target.contiguous()                                # transposed_target --> h w 
                    error = s_pred.view(-1) != transposed_target[t].view(-1)                        # error_ h w 
                    f_supervision = torch.where(error, 1, 0)
                    f_supervision = f_supervision.view(-1,height,width)                             # f_supervision_ 1 h w
                    supervision[:,t]=f_supervision                                                  # save supervision of total frame 

                    '''
                    "s_pred" is label prediction result of STM. 
                    "error" is case of failure prediction of STM
                    "f_supervision" is filled with 1 in case of error 
                    '''    

            #### Model : observer net
                observer_output= model(pert_images, mem_keys, mem_values, torch.tensor([num_objects]),feats,m4,r3e,r2e)

                # Save the output of model and label into the chunck.
                pred_chunck[:,chunck_counter]=observer_output 
                label_chunck[:,chunck_counter]=f_supervision 

                # Save the observer output for total frame output  
                obs_total_logit[:,t]=observer_output.cpu().detach()                                         #observer_output -> 1 h w 
                chunck_counter+=1
        
        #### Calculate the loss and go backward. 
            loss = args.criterion(pred_chunck.view(-1), label_chunck.float().to(args.device).view(-1))      # loss between observer pred and supervision( failure of STM prediction )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            vid_avg_loss+=loss.cpu().item()                                                                 # add the each chunck loss for calculating average loss. 

            # Manage the cuda memory
            gc.collect()
            torch.cuda.empty_cache() 

    #### set total result variables for accuracy calculating.                
        stm_pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
        stm_pred = torch.tensor(stm_pred)                                                                   # frame h w , STM prediction of whole frame 
    
    #### Calculate the accuracy and average loss. 
        stm_acc += stm_pred.view(-1).eq(transposed_target.contiguous().view(-1)).sum()                      # Calculate the number of pixels that predict correct answer. (STM)  
        obsnet_acc += torch.round(torch.sigmoid(obs_total_logit)).view(-1).eq(supervision.view(-1)).sum()   # Calculate the number of pixels that predict correct answer. (Observer) 
        vid_avg_loss/=num_frames
        avg_loss += vid_avg_loss
        print("Video's avg loss : {}".format(vid_avg_loss))
        
       
       
    avg_loss /= len(Trainloader)                        # devide by number of video. It means average loss. 
    obsnet_acc = 100 * (obsnet_acc / nb_sample)         # devide by number of all pixel. It means accuracy of observer net.
    stm_acc = 100 * (stm_acc / nb_sample)               # devide by number of all pixel. It means accuracy of stm net.



    print(f"\rEpoch Summary: Train Avg loss: {avg_loss:.4f}, "
          f"ObsNet acc: {obsnet_acc:.2f}, "
          f"StmNet acc: {stm_acc:.2f}"
          )

    return avg_loss, obsnet_acc
