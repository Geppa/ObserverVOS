import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import cv2
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import random
from torch.utils import data
import torch.optim.lr_scheduler as scheduler
from torch.utils.tensorboard import SummaryWriter
from Utils.utils import reinitialize
from model.observer import Observer
from train import *
from evaluation import evaluate
from Datasets.obstm_dataset import DAVIS_MO_Train
from Datasets.obstm_dataset import DAVIS_MO_Test
from model.stm_model import *
def main(args):
   
####### initial Dataset setting  
   
    writer = SummaryWriter(args.tboard)
    writer.add_text("option", str(args), 0)
    args.g = '1'
    args.s = 'train'
    args.y = 17
    ROOT=os.getcwd()
    DATA_ROOT='\\Users\\geppa\\Desktop\\STM\\obstm\\DAVIS'
    GPU=args.g
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())
    Trainset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(17,'train'), single_object=False)
    Trainloader = data.DataLoader(Trainset, batch_size=1, num_workers=1,shuffle = True, pin_memory=True)
    Valset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(17,'val'), single_object=False)
    Valloader = data.DataLoader(Valset, batch_size=1, num_workers=1,shuffle = True, pin_memory=True)
    Evalset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(17,'evaluation'), single_object=False)
    Evalloader = data.DataLoader(Evalset, batch_size=1, num_workers=1,shuffle = True, pin_memory=True)

#### Model init
    model=Observer()
    model = torch.nn.DataParallel(model).cuda()

    
    #set optim


    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = scheduler.MultiStepLR(optimizer, milestones=[args.epoch // 2, args.epoch-5], gamma=0.2)

    
##### Original Space Time Network setting
    stan_stm = STM().to(args.device)
    state_dict = torch.load(args.stm_file)              # Stm weight load. 
    
    # create new OrderedDict that does not contain `module.` --> Using this codes because of environment issue. 
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]                                    # remove `module.`
        new_state_dict[name] = v
    
    # load params
    stan_stm.load_state_dict(new_state_dict)
    stan_stm.eval()        

   
##### Main start.    
    if args.train_check==0: #if not train. 
        print('no_train mode')
        model.load_state_dict(torch.load(os.path.join(args.obsnet_file, "best.pth"))) 
        model.eval()
        evaluate(0,stan_stm, model, Valloader, "Test", writer, args) 
    else: #if train.
        print('epoch start')
        best = 100 # Variable for saving best model 
        start1 = time.time()

        for epoch in range(0, args.epoch+1):
            start2 = time.time()
            print(f"######## Epoch: {epoch} || Time: {(time.time() - start1)/60:.2f} min ########")             #calculate time of epoch process
            train_loss, obsnet_acc = training(epoch, stan_stm,model, Trainloader, optimizer, writer, args)      # model train.
            val_loss, results_obs = evaluate(epoch, stan_stm, model, Evalloader, "evaluation", writer, args)    # model evaluate for validation. 
            print("time of 1 epoch:{}".format(time.time()-start2))                                              #calculate time of one epoch.  

            #save every epoch
            model_to_save = model.module.state_dict()
            torch.save(model_to_save, os.path.join(args.obsnet_file, f"epoch{epoch:03d}.pth"))
            
            # Save Best model
            if val_loss < best:                                                                                 
                print("save best net!!!")
                best = val_loss
                model_to_save = model.module.state_dict()
                torch.save(model_to_save, os.path.join(args.obsnet_file, "best.pth"))
                
                # memorize best epoch
                best_epoch = epoch                                                                               

                # When the process ends, show what epoch is best. 
                if epoch == args.epoch:
                    print("Finally, {}'s model is saved".format(best_epoch)) 
            
            sched.step() 
    writer.close()

# weight root 
OBSTMROOT=os.path.join(os.getcwd(),'obstm_pth')
stm_path =os.path.join(os.getcwd(),'STM_weights.pth')


if __name__ == '__main__':
    ### Argparse ### 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_check",   type=int,   default=1,       help="train check")
    parser.add_argument("--dset_folder",   type=str,   default="",       help="path to dataset")
    parser.add_argument("--stm_file",   type=str,   default=stm_path,       help="path to stm")
    parser.add_argument("--obsnet_file",   type=str,   default=OBSTMROOT,help="path to obsnet")
    parser.add_argument("--data",          type=str,   default="",       help="CamVid|StreetHazard|BddAnomaly")
    parser.add_argument("--tboard",        type=str,   default="\\Users\\geppa\\Desktop\\server_back\\obstm\\tensorboard", help="path to tensorboeard log")
    parser.add_argument("--model",         type=str,   default="Observer", help="observer") ## help not fixed
    parser.add_argument("--optim",         type=str,   default="SGD",    help="type of optimizer SGD|AdamW")
    parser.add_argument("--T",             type=int,   default=50,       help="number of forward pass for ensemble")
    parser.add_argument("--seed",          type=int,   default=-1,       help="seed, if -1 no seed is use")
    parser.add_argument("--bsize",         type=int,   default=1,        help="batch size")
    parser.add_argument("--lr",            type=float, default=2e-2,     help="learning rate of obsnet")
    parser.add_argument("--Temp",          type=float, default=1.2,      help="temperature scaling ratio")
    parser.add_argument("--noise",         type=float, default=0.,       help="noise injection in the img data")
    parser.add_argument("--epsilon",       type=float, default=0.1,      help="epsilon for adversarial attacks")
    parser.add_argument("--gauss_lambda",  type=float, default=0.002,    help="lambda parameters for gauss params")
    parser.add_argument("--epoch",         type=int,   default=20,       help="number of epoch")
    parser.add_argument("--num_workers",   type=int,   default=1,        help="number of workers")
    parser.add_argument("--num_nodes",     type=int,   default=1,        help="number of node")
    parser.add_argument("--adv",           type=str,   default="max_uncertainty_patch",   help="type of adversarial attacks")
    #parser.add_argument("--adv",           type=str,   default="max_neighbor_patch",   help="type of adversarial attacks")
    #parser.add_argument("--adv",           type=str,   default="max_random_patch",   help="type of adversarial attacks")
    #parser.add_argument("--adv",           type=str,   default="max_dlt_class_patch",   help="type of adversarial attacks")
    parser.add_argument("--test_multi",    type=str,   default="obsnet", help="test all baseline, split by comma")
    parser.add_argument("--drop",          action='store_true',          help="activate dropout in segnet")
    parser.add_argument("--no_img",        action='store_true',          help="use image for obsnet")
    parser.add_argument("--resume",        action='store_true',          help="restart the training")
    parser.add_argument("--obs_mlp",       action='store_true',          help="use a smaller archi for obsnet")
    parser.add_argument("--test_only",     action='store_true',          help="evaluate methods")
    parser.add_argument("--no_residual",   action='store_true',          help="remove residual connection for obsnet")
    parser.add_argument("--no_pretrained", action='store_true',          help="load segnet weight for the obsnet")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc")
    parser.add_argument("-s", type=str, help="set")
    parser.add_argument("-y", type=int, help="year")
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    args = parser.parse_args()

    # Setting multi GPUs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #setting dataset detail.
    args.size = [480, 960]  #will be set in train/eval code. 
    args.crop = (150, 250)
    args.pos_weight = torch.tensor([3]).to(args.device)
    args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
    args.patch_size = [300, 360, 160, 200]
    args.mean = [0.485, 0.456, 0.406]
    args.std = [0.229, 0.224, 0.225]
    args.nclass = 0       # will be set in train/eval code. 

    #for plot
    args.cmap = {
    
        0: (128, 128, 128),  # sky
        1: (128, 0, 0),      # building
        2: (192, 192, 128),  # column_pole
        3: (128, 64, 128),   # road
        4: (0, 0, 192),      # sidewalk
        5: (128, 128, 0),    # Tree
        6: (192, 128, 128),  # SignSymbol
        7: (64, 64, 128),    # Fence
        8: (64, 0, 128),     # Car
        9: (64, 64, 0),      # Pedestrian
        10: (0, 128, 192)}  # Anomaly
    args.class_name = ["Background", "1", "2", "3", "4", "5", "6",
                "7", "8", "9", "10"]
    
    args.one = torch.FloatTensor([1.]).to(args.device)
    args.zero = torch.FloatTensor([0.]).to(args.device)

    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        random.seed(args.seed)

    args.test_multi = args.test_multi.split(",")
    
    main(args)
