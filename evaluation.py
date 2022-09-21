import random
import torch
from torchvision.utils import make_grid
import numpy as np
from Utils.affichage import draw, plot
from Utils.utils import entropy, transform_output
from Utils.utils import monte_carlo_estimation, mcda_estimation, gauss_estimation, ensemble_estimation, odin_estimation
from Utils.metrics import print_result
import torch.nn.functional as F
from Utils.utils import memory_save
from Utils.utils import stm_segment
from Utils.adv_attacks import * 


def evaluate(epoch, stan_stm, model, loader, split, writer, args):
    """ Evaluate method contain in the arguments args.test_multi
        epoch  ->  int: current epoch
        split  ->  str: Test or Val
        loader ->  torch.DataLoader: the dataloader
        obsnet ->  torch.Module: the observer to test
        segnet ->  torch.module: the stm pretrained and freeze
        writer ->  SummaryWriter: for tensorboard log
        args   ->  Argparse: global parameters
    return:
        avg_loss     -> float: average loss on the dataset
        results_obs  -> dict: the result of the obsnet on different metrics
    """

#### Print start training the video.

    YEAR = args.y
    SET = args.s
    MODEL = 'STM_Observer'
    code_name = '{}_DAVIS_{}_{}'.format(MODEL,YEAR,SET)
    print('Start Evaluating:', code_name)

##### Evaluation

    model.eval()
    avg_loss,total_obsnet_acc, total_segnet_acc = 0, 0., 0.     #Set values for using globally
    #r = random.randint(0, len(loader) - 1)                     -> not using in normal code. 
    
    for i, V in enumerate(loader):

    #### Variable preprocessing 
        Fs, Ms, num_objects, info = V
        seq_name = info['name'][0]
        num_frames = info['num_frames'][0].item()
        print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
        args.nclass = num_objects
        
        # set variables for loss and acc. 
        vid_avg_loss=0.0
        obsnet_acc,segnet_acc,nb_sample=0.,0.,0 
        
        #convert to Observer Variables from STM variables. (to images)
        images = torch.transpose(Fs,3,4)        #images[batch,channel,frame,w,h]
        Ms_2 = Ms[0]                            #-> class(11)x frame x h x w 
        Ms_2 = torch.transpose(Ms_2,3,2)        #-> class(11) x frame x w x h
        Ms_2 = torch.argmax(Ms_2,dim=0)         #-->frame x w x h
        target = Ms_2

        #Setting for observer net variables
        bsize, channel,frame, width, height = images.size()
        args.size=[width,height]
        images = images.to(args.device)
        target = target.to(args.device)
        transposed_target=torch.transpose(target,1,2)
       
        #Setting for STM net variable
        Es = torch.zeros_like(Ms)
        Es[:,:,0] = Ms[:,:,0]

        #### calculate the memory keys and values with stm inference about current video. it needs for adv attack method. 
        with torch.no_grad():
            mem_keys, mem_values,stm_seg,stm_Es = memory_save(stan_stm,Fs,Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
        

    #### Evaluate process on.     
        for t in range(1, num_frames):

        #### Make ood data for evaluate the observer net's accuracy
            if args.adv != "none":                                                             # perform the LAA
                images[:,:,t], mask = select_attack(images[:,:,t].to(args.device), target[t].to(args.device),stan_stm,V, args,mem_keys,mem_values,t,stm_seg,stm_Es)
                    
            with torch.no_grad():
                nb_sample += bsize * height * width

                #Setting for using in STM. 
                pert_images=images[:,:,t]
                pert_images=torch.transpose(pert_images,2,3) #1 3 h w

            #### STM segment inference inserting attacked input.  
                _,Es,logit,m4,r3e,r2e,feats = stm_segment(stan_stm,Fs,pert_images,Ms, num_frames, num_objects,mem_keys,mem_values,t, Mem_every=5, Mem_number=None)

                #Set output to use in loss calculating and accuracy calculating 
                pred = np.argmax(Es[0,:,t].cpu().numpy(), axis=0).astype(np.uint8)                  #ES[0,:,t]-> class h w 
                s_pred=torch.Tensor(pred).contiguous()                                              # h w
                transposed_target=transposed_target.contiguous()                                    #h w 
    
                #model error detection
                supervision = s_pred.view(-1).to(args.device) != transposed_target[t].view(-1)      #s_pred : h w , transposed_target[t] : h w 
                supervision = torch.where(supervision, args.one, args.zero).to(args.device)         #supervision : h*w, cuz s_pred and transposed_target[t] is flattened
                supervision = supervision.view(bsize,-1)                                            #supervision : h*w batchsize

                '''
                "s_pred" is label prediction result of STM. 
                "supervision" is filled with 1 in case of failure prediction of STM
                '''    

                #set variabls to use
                pred = s_pred.float().view(-1, 1).to(args.device)                                   #(h*w, 1) 
                target = transposed_target[t].float().view(-1, 1)                                   #(h*w, 1)

            #### use if for implementation other uncertainty measure. 
                if "obsnet" in args.test_multi:  # our method

                    observer_output = model(pert_images, mem_keys, mem_values, torch.tensor([num_objects]),feats,m4,r3e,r2e)    #batch pressed_class(1) h w 
                    observer_output=observer_output.to(args.device)
                    obs_pred = transform_output(pred=observer_output, bsize=bsize, nclass=1)                                    #  obs_pred -> _,1 
                    loss = args.criterion(obs_pred.view(-1), supervision.view(-1))                                              # loss between observer pred and supervision( failure of STM prediction )
            
                #### Calculate the accuracy and average loss. 
                    segnet_acc += pred.view(-1).eq(target.view(-1)).sum()                                                       # Calculate the number of pixels that predict correct answer. (STM) 
                    obsnet_acc += torch.round(torch.sigmoid(obs_pred)).view(-1).eq(supervision.view(-1)).sum()                  # Calculate the number of pixels that predict correct answer. (Observer) 
                    vid_avg_loss += loss.cpu().item()

                    if (i == 0) & (t == 14):
                        save = np.savetxt('segnetAcc_pred.txt', pred.detach().cpu(), delimiter=',')
                        save = np.savetxt('segnetAcc_target.txt', target.detach().cpu(), delimiter=',')
                        save = np.savetxt('obsnetAcc_sigmoid.txt', torch.sigmoid(obs_pred).detach().cpu(), delimiter=',')
                        save = np.savetxt('obsnetAcc_pred.txt',obs_pred.detach().cpu(), delimiter=',')
                        save = np.savetxt('obsnetAcc_pred_round_sigmoid.txt', torch.round(torch.sigmoid(obs_pred).detach().cpu()), delimiter=',')
                        save = np.savetxt('obsnetAcc_supervision.txt', supervision.detach().cpu(), delimiter=',')

                    obs = torch.sigmoid(obs_pred)
                    batch_res_obs = torch.cat((obs.view(-1, 1), pred, target), dim=1)
                    res_obs = batch_res_obs.cpu() if i == 0 else torch.cat((res_obs, batch_res_obs.cpu()), dim=0)               # metric result of observer network 
                
                    print(f"\rEval loss: {loss.cpu().item():.4f}, "
                        f"Progress: {100 * (i / len(loader)):.2f}%",
                        end="\r")
                 

    #### Calculating result of video-wise loss and accuracy
        vid_avg_loss /= num_frames
        print("Video_Average_loss : {} \r".format(vid_avg_loss))

        avg_loss+=vid_avg_loss
        vid_obsnet_acc = 100 * (obsnet_acc / nb_sample)
        vid_segnet_acc = 100 * (segnet_acc / nb_sample)
        total_obsnet_acc += vid_obsnet_acc
        total_segnet_acc += vid_segnet_acc
        print("Video_Average_obsnet_acc : {} \r".format(vid_obsnet_acc))
        print("Video_Average_segnet_acc : {} \r".format(vid_segnet_acc))

#### Calculating result of total loss and accuracy
    res_obs = print_result("ObsNet", "Val", res_obs, writer, epoch, args)

    avg_loss /= len(loader)
    total_obsnet_acc /= len(loader)
    total_segnet_acc /= len(loader)
    writer.add_scalars('data/' + split + 'Loss', {"loss": avg_loss}, epoch)

    print(f"\rEpoch Summary: {split} Avg loss: {avg_loss:.4f}, "
          f"ObsNet acc: {total_obsnet_acc:.2f}, "
          f"SegNet acc: {total_segnet_acc:.2f}\t"
          )

    return [avg_loss, res_obs] if "obsnet" in args.test_multi else avg_loss
