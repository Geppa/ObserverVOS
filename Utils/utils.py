import copy
import glob
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
from Datasets.seg_transfo import AdjustGamma, AdjustSaturation, AdjustHue, AdjustBrightness, AdjustContrast, \
    SegTransformCompose

'''
def transform_output(pred, bsize, nclass):
    output = pred.view(bsize, nclass, -1)
    output = torch.transpose(output, 1, 2).contiguous()
    return output.view(-1, nclass)
'''

def transform_output(pred, bsize, nclass):
    output = pred.view(bsize, nclass, -1)
    output = torch.transpose(output, 1, 2).contiguous()
    return output.view(-1, nclass)


def reinitialize(model, ref_model):
    """ Initialize the weight of the observer network with those of the segnet
        model     -> nn.Module: the model which will be reinitialize
        ref_model -> Path: path where to find the wieghts"""
    ref_param = torch.load(ref_model)
    new_params = model.state_dict().copy()
    for indice, k in enumerate(ref_param.keys()):
        if indice == len(ref_param.keys())-7:  # remove incompatible layer
            break
        if k in new_params.keys():
            new_params[k] = ref_param[k]
    model.load_state_dict(new_params)


def entropy(pred, softmax=False):
    """ Compute the entropy of the prediction
        pred    -> Tensor: (b x w x h x n_class) the ouput of the model
        softmax -> Bool: apply softmax on pred or not
    return:
        entropy -> Tensor: (b x 1) the entropy """
    p = torch.softmax(pred, -1) if softmax else pred
    log_p = torch.log_softmax(pred, -1) if softmax else torch.log(pred)
    return -torch.sum(p * log_p, dim=-1)


def monte_carlo_estimation(segnet, images, bsize, width, height, args):
    """ Compute the mean of T forward passes with dropout corresponding to MCDropout"""
    mc = torch.zeros((bsize, width * height, args.nclass)).to(args.device)
    for t in range(args.T):
        output_m = segnet(images, mc_drop=True)[-1]
        output = torch.transpose(output_m.view(bsize, args.nclass, -1), 1, 2).contiguous()
        mc += torch.softmax(output, -1)
    return mc / args.T


def mcda_estimation(segnet, images, bsize, width, height, args):
    """Compute the mean of T forward passes with random transformation corresponding to MCDA"""

    def augmented_img(x, device, transfo):
        """ Apply transforation in the image """
        imgs = []
        for image in x:
            img = image.cpu().numpy()
            img = ((img - img.min()) / (img.max() - img.min()))
            img = np.transpose(img, (1, 2, 0)) * 255
            img = Image.fromarray(np.uint8(img))

            img, _ = transfo(img, None)
            img = tf.to_tensor(img)
            imgs.append(img.unsqueeze(0))
        return torch.cat(imgs).to(device)

    transfo = SegTransformCompose(AdjustGamma(0.3), AdjustSaturation(0.1), AdjustHue(0.03),
                                  AdjustBrightness(0.2), AdjustContrast(0.2))
    uncertainty = torch.zeros((bsize, width * height, args.nclass)).to(args.device)
    for t in range(int(args.T / 2)):
        output_m = segnet(augmented_img(images, args.device, transfo))
        output = torch.transpose(output_m.view(bsize, args.nclass, -1), 1, 2).contiguous()
        uncertainty += torch.softmax(output, -1)
    uncertainty /= (args.T / 2)
    return uncertainty


def gauss_estimation(segnet, images, bsize, width, height, args):
    """ Compute the mean of T forward passes with noise in the params corresponding to Gauss Pert """
    uncertainty = torch.zeros((bsize, width * height, args.nclass)).to(args.device)
    tmp_net = copy.deepcopy(segnet)
    for t in range(int(args.T)):
        tmp_net.module.load_state_dict(torch.load(args.segnet_file))
        for _, param in enumerate(tmp_net.module.parameters(), 1):
            param.add_(torch.randn(param.size(), device=args.device) * args.gauss_lambda)
        output_m = tmp_net(images, mc_drop=False)
        output = torch.transpose(output_m.view(bsize, args.nclass, -1), 1, 2).contiguous()
        uncertainty += torch.softmax(output, -1)
    uncertainty /= args.T
    return uncertainty


def ensemble_estimation(segnet, images, bsize, width, height, args):
    """ Compute the mean of M models predictions corresponding to Deep Ensemble
        All networks should be in the same folder with the same prefixe than the main segnet i.e.:
            ./segnet_1.pth, ./segnet_2.pth, ./segnet_3.pth, ./segnet_4.pth, etc.
    """
    uncertainty = torch.zeros((bsize, width * height, args.nclass)).to(args.device)
    tmp_net = copy.deepcopy(segnet)
    tmp_net.eval()
    glob_file = glob.glob(args.segnet_file[:-4] + "*")
    for indice, i in enumerate(glob_file):
        tmp_net.load_state_dict(torch.load(i))
        output_m = tmp_net(images, mc_drop=False)[-1]
        output = torch.transpose(output_m.view(bsize, args.nclass, -1), 1, 2).contiguous()
        uncertainty += torch.softmax(output, -1)

    uncertainty /= len(glob_file)
    return uncertainty


def odin_estimation(segnet, images, bsize, args):
    """ ODIN estimation """
    # perform the adversarial attack
    odin_images = images.clone()
    odin_images.requires_grad = True
    odin_pred = segnet(odin_images, mc_drop=False)
    odin_pred = transform_output(odin_pred, bsize, args.nclass)
    odin_target = torch.softmax(odin_pred, -1).max(-1)[0]                    # prediction

    loss = F.cross_entropy(odin_pred, odin_target.reshape(-1).long())
    segnet.zero_grad()
    loss.backward()                                                          # Compute the loss

    data_grad = odin_images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed = args.epsilon * sign_data_grad
    odin_images.data -= perturbed
    odin_images.detach()

    odin_logit = segnet(odin_images, mc_drop=False)                             # New pred on the attacked images
    odin_pred = transform_output(odin_logit, bsize, args.nclass).detach()
    uncertainty = 1 - torch.softmax(odin_pred / args.Temp, -1).max(-1)[0]       # 1 - softmax with Temperature Scaling
    return uncertainty


### for saving in memory. it needs in adv attack.  
def memory_save(stan_stm,Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    #for t in tqdm.tqdm(range(1, num_frames)):
    for t in range(1,num_frames):
           # memorize
        with torch.no_grad():
            prev_key, prev_value = stan_stm(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        
        # segment
        with torch.no_grad():
            logit,_,_,_,_  = stan_stm(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    #return memory keys and values
    return this_keys,this_values,pred,Es[0]




def stm_segment(stan_stm,Fs,images_stm, Ms, num_frames, num_objects,mem_keys,mem_values,t, Mem_every=None, Mem_number=None):
   
    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    if(Fs[:,:,0].shape!=images_stm.shape):
        images_stm=torch.transpose(images_stm,2,3)
    #print(mem_keys.shape) #1 11 128 13 30 54
    #print(mem_values.shape) #1 11 512 13 30 54

    memory_index=t//Mem_every
    this_keys = mem_keys[:,:,:,:memory_index]
    this_values = mem_values[:,:,:,:memory_index]

    logit,m4,r3e,r2e,feats = stan_stm(images_stm, this_keys, this_values, torch.tensor([num_objects]))   #############
    sf_logit= F.softmax(logit, dim=1)
    #print("sf_logit'shape:{}".format(sf_logit.shape))
    ##
    pred_class=sf_logit[0]
    Es[:,:,t] = F.softmax(logit, dim=1)
    return pred_class,Es,logit,m4,r3e,r2e,feats
