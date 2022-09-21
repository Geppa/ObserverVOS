import random
import torch
import torch.nn.functional as F
from skimage.draw import random_shapes
import numpy as np
from Utils.utils import transform_output
import tqdm
import cv2
import os
import math
import tensorflow
from torch.autograd import Variable
from Utils.utils import stm_segment
from torchviz import make_dot


def add_shape(args):
    """ return a Tensor with a random shape
        1 for the shape
        0 for the background
        The size is controle by args.patch_size
    """
    #args.patch_size[0:2]  --> patch_size[0],patch_size[1]
    image, _ = random_shapes(args.patch_size[0:2], min_shapes=6, max_shapes=10,
                             intensity_range=(0, 50),  min_size=args.patch_size[2],
                             max_size=args.patch_size[3], allow_overlap=True, num_channels=1)
    image = torch.round(1 - torch.FloatTensor(image)/255.)
    return image.squeeze().to(args.device)

def adaptive_add_shape(args,patch_size):
    """ return a Tensor with a random shape
        1 for the shape
        0 for the background
        The size is controle by args.patch_size
    """
    #args.patch_size[0:2]  --> patch_size[0],patch_size[1]
    image, _ = random_shapes(patch_size[0:2], min_shapes=6, max_shapes=10,
                             intensity_range=(0, 50),  min_size=patch_size[2],
                             max_size=patch_size[3], allow_overlap=True, num_channels=1)
    image = torch.round(1 - torch.FloatTensor(image)/255.)
    return image.squeeze().to(args.device)


###segnet-->stan_stm
def fgsm_attack(images, target, V,stan_stm, mask, mode, args,mem_keys,mem_values,t):
    """ Perform a FGSM attack on the image
        images -> Tensor: (b,c,w,h) the batch of the images
        target -> Tensor: the label
        stan_stm-> torch.Module: The segmentation network
        mask   -> Tensor: (b,1,w,h) binary mask where to perform the attack
        mode   -> str: either to minimize the True class or maximize a random different class
        args   ->  Argparse: global arguments
    return
        images       -> Tensor: the image attacked
        perturbation -> Tensor: the perturbation
    """
    bsize, channel, width, height = images.size()
    images += 0.03 * torch.randn_like(images) * mask     # add some noise for more diverse attack
    images.requires_grad = True
   
    #init stm inputs
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    images_stm=images

    stm_pred,_,_,_,_,_,_ = stm_segment(stan_stm,Fs,images_stm,Ms, num_frames, num_objects,mem_keys,mem_values,t, Mem_every=5, Mem_number=None) #class h w 
    

    c,h,w=stm_pred.shape
    
    stm_pred = transform_output(stm_pred, bsize, 11).to(args.device)#stm_pred 구조 변경
    
    #print("flatten stm_pred with class: {}".format(stm_pred.shape))
    args.nclass=int(args.nclass)###################
    #print(args.nclass)
    if mode == "max":
        #print(args.nclass+1)
        
        if args.nclass==1:
            fake_target=torch.randint(2,(h,w)).to(args.device)
        else:
            fake_target=torch.randint(args.nclass,(h,w)).to(args.device)
        
        #fake_target = torch.randint(0,args.nclass, size=(bsize, 1)).expand_as(target).to(args.device)
        #fake_target = torch.randint(args.nclass - 1, size=(bsize, 1)).expand_as(target).to(args.device)
       # print("fake_target after randint: {}".format(fake_target.shape))
        #fake_target=torch.transpose(fake_target,0,2).contiguous()
        #print(fake_target.shape)
        #fake_target=fake_target.view(-1,args.nclass)
        #fake_target=torch.sum(fake_target,dim=0)
        fake_target=fake_target.view(-1,)
        #print("1213213 : {} ".format(stm_pred.shape))
        #print(fake_target.shape)
        #print(stm_pred.get_device())
        #print(fake_target.get_device())
        #print(stm_pred.cpu())
        #print(fake_target)

        
        loss = F.cross_entropy(stm_pred, fake_target)


        '''if args.nclass==1:
            loss=F.binary_cross_entropy(stm_pred, fake_target.float())
        else:
            loss = F.cross_entropy(stm_pred, fake_target)    
        '''
        #loss = F.cross_entropy(stm_pred, fake_target.float())
    elif mode == "min":
        loss = F.cross_entropy(stm_pred, target.long().reshape(-1))
    loss.requires_grad_(True)
    stan_stm.zero_grad()
    #os.environ["PATH"]+=os.pathsep+'C:/Program Files/Graphviz/bin/'
    #make_dot(loss, params=dict(stan_stm.named_parameters())).render(f"graph", format="png")
    loss.backward()
    #print(images.grad.data)
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()

    sign_data_grad *= mask
    if mode == "max":
        perturbed = - args.epsilon * sign_data_grad
    elif mode == "min":
        perturbed = args.epsilon * sign_data_grad

    #print("^^^^^^^^^^^^^^^^^^^^^^^^image shape in last LAA before perturbation : {}".format(images.shape))
    images.data = torch.clamp(images + perturbed, -3, 3)
    #print("^^^^^^^^^^^^^^^^^^^^^^^^image shape in last LAA after perturbation : {}".format(images.shape))

    return images.detach(), perturbed


def generate_mask_all_images(images, args):
    return torch.ones_like(images)


def generate_mask_pixels_sparse(images, args):
    """ generate a mask to attack random pixel on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    return torch.where(torch.rand_like(images) < 0.05, args.one, args.zero)


def generate_mask_class(images, target, stm_pred, args):

    ############ target이 argmax되어서, w h 2dim이고, 따라서 이 코드의 클래스 와이즈는 작동 안할 가능성이 높음. 

    """ generate a mask to attack a random class on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width * height).to(args.device)
    for i in range(len(mask)):
        valid_classes, count = torch.unique(target[i], return_counts=True)
        chosen_classes = valid_classes[random.randint(0, len(valid_classes)-1)]
        mask[i][target[i].view(-1) == chosen_classes] = 1
        
        #######################################################
        if args.adv.startswith("min"):   # do not attack if the sample are already badly classify
            mask[i][target[i].view(-1) != stm_pred[i].view(-1)] = 0
    return mask.view(bsize, 1, width, height).expand_as(images)


def generate_mask_square_patch(images, args):
    """ Generate a mask to attack a random square patch on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).to(args.device)
    for i in range(len(mask)):
        x = random.randint(0, width - args.patch_size[0])
        y = random.randint(0, height - args.patch_size[1])
        h = args.patch_size[1]
        w = args.patch_size[0]
        mask[i, x:x + w, y:y + h] = 1.
    return mask.view(bsize, 1, width, height).expand_as(images)


def generate_mask_random_patch(images, args):
    """ Generate a mask to attack a random shape on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width, height).to(args.device)
    for i in range(len(mask)):
        x = random.randint(0, width - args.patch_size[0])
        y = random.randint(0, height - args.patch_size[1])
        h = args.patch_size[1]
        w = args.patch_size[0]
        shape = add_shape(args)
        mask[i, x:x + w, y:y + h] = shape == 1.
    return mask.view(bsize, 1, width, height).expand_as(images)




################################################################################################################################################


#target shape : Width x Height 

def generate_mask_dlt_class(images, target, stm_pred, args,num_objects):
    """ generate a mask to attack a random class on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask_sum = torch.zeros(width,height)
    
    for i in range(1,num_objects+1):
        class_filter = torch.full((bsize,width,height),i).to(args.device) # 타겟과 비교를 위한 텐서 생성 (해당 레이블로 꽉찬 텐서 생성) 
        flt_target_classify = target.view(-1) == class_filter.view(-1) #타겟과 생성 텐서를 비교해서 겹치는게 있는지 확인
        object_region = torch.where(flt_target_classify, args.one, args.zero) #겹치는게 있으면 1로 초기화, 랜덤 패치(mask)가 1로 설정되기 때문. 
        
        check_empty_region = object_region.cpu().detach().numpy()  #If Object is deleted in video sequence, do not make patch about that object.
        if check_empty_region == torch.zeros((bsize,width,height)).view(-1).to(args.device):
            continue

        object_region = object_region.view(width,height).detach().cpu()
  
    # count mask area pixels.
        adap_mask_flag=True

        if adap_mask_flag==True:
            area_mask = torch.count_nonzero(object_region)
            size_filter = max(1,int(math.sqrt(((area_mask) * (10/100)))))
            #print(size_filter)
        else:
            size_filter=30

    
#size_filter = int((area_mask / area_whole) * 100)
#area_whole = width * height


       # print(object_region.shape)
        object_region=np.uint8(object_region)
        mrp_rect=cv2.getStructuringElement(cv2.MORPH_RECT,(size_filter,size_filter))
        dilated_mask=cv2.dilate(object_region,mrp_rect)
        dilated_mask=torch.Tensor(dilated_mask)
        mask_sum = mask_sum+dilated_mask

    zero_image=torch.zeros(width,height)
    check_mask = mask_sum.view(-1) !=zero_image.view(-1) #모든 클래스에 대한 마스크를 다 더한다. 그것을 제로텐서와 비교해서
    mask_result = torch.where(check_mask,1,0) # 틀린애들은 마스크가 있는 지역이니, 다 1로 초기화해준다. ( 겹치는 마스크들을 하나로 )


####viz
    #print(mask_result.shape)
    writing_dilated_mask=np.transpose(mask_result.view(width,height).numpy(),(1,0))
    cv2.imwrite("\\Users\\geppa\\Desktop\\server_back\\obstm\\dlt_patch\\patch.png",writing_dilated_mask*255)
    #os.system("pause")

    mask_result=mask_result.to(args.device)
    return mask_result.view(bsize, 1, width, height).expand_as(images).to(args.device)   #rgb채널로 만들어줘야 노이즈 가시화가 되니까?


def generate_mask_neighbor_random_patch(images, target, stm_pred, args,num_objects):
    """ generate a mask to attack a random class on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
    return:
        mask -> the mask where to perform the attack """
    
#### 랜덤 패치를 num_obj 개만큼 생성, target에서 얻어온 픽셀의 클래스 정보에 따라, 클래스 지역들 분리. 
#### 만들어진 랜덤 패치가 각 클래스 지역에 무조건 포함되도록 설정. 아니면 재생성. 아마 while문으로 반복?

    bsize, channel, width, height = images.size()
    mask_sum = torch.zeros(width,height).to(args.device)
    
    for i in range(1,num_objects+1):
        class_filter = torch.full((bsize,width,height),i).to(args.device) # 타겟과 비교를 위한 텐서 생성 (해당 레이블로 꽉찬 텐서 생성) 
        flt_target_classify = target.view(-1) == class_filter.view(-1) #타겟과 생성 텐서를 비교해서 겹치는게 있는지 확인
        class_loc= torch.where(flt_target_classify, args.one, args.zero) #겹치는게 있으면 1로 초기화, 랜덤 패치(mask)가 1로 설정되기 때문. 
        
        check_empty_class = class_loc.cpu().detach().numpy()  #If Object is deleted in video sequence, do not make patch about that object.
        if check_empty_class == torch.zeros((bsize,width,height)).view(-1).to(args.device): #If Object is deleted in video sequence, do not make patch about that object.
            continue


        whole_region = width * height #전체 픽셀
        object_region = torch.count_nonzero(class_loc) #각 클래스 오브젝트 픽셀
        object_ratio = object_region/whole_region 
        #패치 크기를 오브젝트의 크기에 비례하게 만들어주기 위함. 기존의 랜덤 패치는 전체 픽셀에 대비해서 고려되었으니
        #기존 랜덤 패치 사이즈에 ratio를 적용해주면 오브젝트 크기에 비례할 것. 
        
        check_find=0
        check_patch_flag=0
        while check_patch_flag==0:  #For generating the aimed patch. 
            mask = torch.zeros(bsize, width, height).to(args.device)
            #Generate random patch
            for i in range(len(mask)):  # mask의 첫 dim의 size . (=mask.size[0])  --> batch마다 생성해주기 위함임. 
                x = random.randint(0, width - max(20,int(args.patch_size[0]*min(2,object_ratio*10)))) #Object_ratio : 오브젝트 크기 비례하여 패치만듦
                y = random.randint(0, height - max(20,int(args.patch_size[1]*min(1,object_ratio*10)))) # *10 : ratio로 인해 너무 작은 값을 갖지 않도록. 
                h = max(20,int(args.patch_size[1]*min(1,object_ratio*10)))
                w = max(20,int(args.patch_size[0]*min(2,object_ratio*10)))
                
                #print("Finding good patch")
                patch_size=[max(4,w),max(4,h),max(2,int(w*0.6)),max(2,int(h*0.6))] #0.6 --> ratio of patch size
                shape = adaptive_add_shape(args,patch_size)

            mask[i, x:x + w, y:y + h] = shape == 1.
            flt_mask=mask.view(-1)
            check_loc = class_loc * flt_mask #class 위치 정보 텐서와 생성된 patch mask 정보 텐서를 곱함. 1인 픽셀은 목표 리젼과 패치 리젼이 겹치는 픽셀
            check_patch_nonzero = torch.count_nonzero(check_loc) #0이 아닌게 있는지 -> 겹치는게 있는지 -> 값이0이면 겹치는게 없는 거니까 다시 생성 필요.
            if check_patch_nonzero > 0:
                check_patch_flag=1
            else:
                check_find+=1

            if check_find==100: #패치가 어노테이션 리젼에 맞는지 체크하는 과정이 100번을 넘을 경우 스킵.
                check_patch_flag=1

        #masks.append(mask)
        mask_sum = mask_sum+mask.to(args.device)

    zero_image=torch.zeros(width,height).to(args.device)
    check_mask = mask_sum.view(-1) !=zero_image.view(-1) #모든 클래스에 대한 마스크를 다 더한다. 그것을 제로텐서와 비교해서
    mask_result = torch.where(check_mask,args.one,args.zero) # 틀린애들은 마스크가 있는 지역이니, 다 1로 초기화해준다. ( 겹치는 마스크들을 하나로 )


###viz
    writing_mask=np.transpose(mask_result.detach().cpu().view(width,height).numpy(),(1,0))
    cv2.imwrite("\\Users\\geppa\\Desktop\\server_back\\obstm\\nbh_patch\\patch.png",writing_mask*255)
    #os.system("pause")


    return mask_result.view(bsize, 1, width, height).expand_as(images)





def generate_uncertainty_patch(images, target, stm_pred, args,stm_Es):
    """ generate a mask to attack a random class on the images
        images -> Tensor: (b,c,w,h) the batch of images
        args   -> Argparse: global arguments
        stm_Es -> Tensor: (class, h, w])
    return:
        mask -> the mask where to perform the attack """
    bsize, channel, width, height = images.size()
    mask = torch.zeros(bsize, width,height).to(args.device)
   
    stm_Es=torch.transpose(stm_Es,1,2)
    stm_confidence,_=torch.max(stm_Es,dim=0) #w h 
    stm_uncertainty = 1 - stm_confidence
    stm_uncertainty = stm_uncertainty.view(1,1,width,height)

    # size of pooling kernel -> It decides size of attack patch_i  
    kernel_w=int(width/30)
    kernel_h=int(height/30)

    grid=torch.nn.AvgPool2d((kernel_w,kernel_h),stride=1)
    unc_grid=grid(stm_uncertainty)

    #set Threshold of uncertainty 
    max_unc,_=torch.max(unc_grid[0][0].view(-1),dim=0)
    thresh_unc = float(max_unc * 0.1)

    #Choose uncertain indexes with threshold
    thresh_index = np.where(unc_grid[0][0]>thresh_unc)

#   _,_,grid_w,grid_h=unc_grid.shape
 # thresh_grid = torch.where(unc_grid[0][0]>0.006,float(1),float(0))
    #print(max_unc)
    #print(thresh_unc)

    dim1_idx=[]
    dim2_idx=[]
    
    dim1_idx,dim2_idx=thresh_index
    
    # Sum all patch_i to one big patch
    for i in range(len(dim1_idx)):
        mask[:,dim1_idx[i]:dim1_idx[i]+kernel_w,dim2_idx[i]:dim2_idx[i]+kernel_h]=1

###viz
    mask_result=mask
    writing_mask=np.transpose(mask_result.detach().cpu().view(width,height).numpy(),(1,0))
    cv2.imwrite("\\Users\\geppa\\Desktop\\server_back\\obstm\\unc_filter\\patch.png",writing_mask*255)
    #os.system("pause")
    return mask.view(bsize, 1, width, height).expand_as(images)


################################################################################################################################################

def select_attack(images, target, stan_stm,V,args,mem_keys,mem_values,t,stm_seg,stm_Es):
    """ Select the right attack given args.adv """
    #init stm inputs
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()

            
    if args.adv.startswith("min_"):
        if args.adv.endswith("all_image"):
            mask = generate_mask_all_images(images, args)
        elif args.adv.endswith("pixels_sparse"):
            mask = generate_mask_pixels_sparse(images, args)
        elif args.adv.endswith("class"):
            mask = generate_mask_class(images, target, stm_seg, args)
        elif args.adv.endswith("square_patch"):
            mask = generate_mask_square_patch(images, args)
        elif args.adv.endswith("random_patch"):
            mask = generate_mask_random_patch(images, args)
        return fgsm_attack(images, target, stan_stm, mask, "min", args)
    elif args.adv.startswith("max_"):
        if args.adv.endswith("all_image"):
            mask = generate_mask_all_images(images, args)
        elif args.adv.endswith("pixels_sparse"):
            mask = generate_mask_pixels_sparse(images, args)
        elif args.adv.endswith("class"):
            mask = generate_mask_class(images, target, stm_seg, args)
        elif args.adv.endswith("square_patch"):
            mask = generate_mask_square_patch(images, args)
        elif args.adv.endswith("random_patch"):
            mask = generate_mask_random_patch(images, args)
        elif args.adv.endswith("dlt_class_patch"):
            mask = generate_mask_dlt_class(images, target, stm_seg, args,num_objects)
        elif args.adv.endswith("neighbor_patch"):
            mask = generate_mask_neighbor_random_patch(images, target, stm_seg, args,num_objects) 
        elif args.adv.endswith("uncertainty_patch"):
            mask = generate_uncertainty_patch(images, target, stm_seg, args,stm_Es[:,t])   
        return fgsm_attack(images, target,V, stan_stm, mask, "max", args,mem_keys,mem_values,t)
    else:
        raise NameError('Unknown attacks, please check args.adv arguments')


'''

def stm_segment(stan_stm,Fs,images_stm, Ms, num_frames, num_objects,mem_keys,mem_values,t, Mem_every=None, Mem_number=None):
   
    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    if(Fs[:,:,0].shape!=images_stm.shape):
        images_stm=torch.transpose(images_stm,2,3)
    print(mem_keys.shape) #1 11 128 13 30 54
    print(mem_values.shape) #1 11 512 13 30 54

    memory_index=t//Mem_every
    this_keys = mem_keys[:,:,:,:memory_index]
    this_values = mem_values[:,:,:,:memory_index]

    logit = stan_stm(images_stm, this_keys, this_values, torch.tensor([num_objects]))   #############
    sf_logit= F.softmax(logit, dim=1)
    print("sf_logit'shape:{}".format(sf_logit.shape))
    ##
    pred_class=sf_logit[0]
    return pred_class

'''
#stan_stm을 받아서 연산해서 돌려주는 함수 구현 
def stan_Run_video(stan_stm,Fs,images_stm, Ms, num_frames, num_objects,mem_keys,mem_values, Mem_every=None, Mem_number=None):
    # initialize storage tensors

    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    #Es = torch.zeros_like(Ms).to('cuda')
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
        if t<num_frames-1:
            with torch.no_grad():
                logit = stan_stm(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
        elif t==num_frames-1:  ##############
            if(Fs[:,:,t].shape!=images_stm.shape):
                images_stm=torch.transpose(images_stm,2,3)
            # print('check fs grad')   
            #print("@@@@@@@@@@@@@@@@@@@@@@@@@in adv attack in run vid, size of images_stm.{} ".format(images_stm.shape))
            #print("@@@@@@@@@@@@@@@@@@@@@@@@@in adv attack in run vid, size of Fs[:,:,t].{} ".format(Fs[:,:,t].shape))
            
            logit = stan_stm(images_stm, this_keys, this_values, torch.tensor([num_objects]))   #############
            # print('check logit grad')
            # print(logit.grad)
        Es[:,:,t] = F.softmax(logit, dim=1)
        # print('check es grad')
        # print(Es.grad)
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
    #pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    pred = torch.argmax(Es[0],dim=0)

    ##
    pred_class=Es[0]
    return pred_class

    '''
    pred_class=pred.type(torch.uint8)
    stm_pred_class = torch.transpose(pred_class,2,1) #frame X w X h
    print("###########################stm_pred(after transpose)'s shape: {}".format(stm_pred_class.shape))
    stm_pred_class=stm_pred_class[num_frames-1]
    print("STM_pred(query)'s shape:{}".format(stm_pred_class.shape)) #frame x w x h
    return stm_pred_class
    '''
