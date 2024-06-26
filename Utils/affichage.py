import torch
from torchvision.utils import make_grid


def plot(x, output_m, y, args, normalize=True):
    """  Construct new images with the 1st image, the 1st prediction and the 1st GT of the batch
        x          -> Tensor: (b,c,w,h) the batch of the images
        output_m   -> Tensor: the output of the segmentation model
        y          -> Tensor: the segmentation labels
        args       -> Argparse: global arguments
        normalize  -> Bool: normalize
    return:
        images -> Tensor: the images, the segmentation map and the Gt Segmentation """
    print("plot start!!!!!!!!!!!")

    w, h = args.size
    _pred = torch.argmax(output_m, dim=1).cpu()
    prediction = torch.FloatTensor([args.cmap[p.item()] for p in _pred[0].view(-1)])
    #print(prediction)
    #print(prediction.shape)
    prediction = prediction.transpose(0, 1).view(3, h, w) / 255.
    #print(prediction)
    #print("-------------------------------------------------------")
    #print(prediction.shape) # in eval, torch.Size([3, 480, 854])
    y_visu = torch.FloatTensor([args.cmap[p.item()] for p in y.view(-1)])
    #print(y_visu)
    #print(y_visu.shape)
    y_visu = y_visu.transpose(0, 1).view(3, h, w) / 255.
    #print(y_visu)
    #print("-------------------------------------------------------")
    #print(y_visu.shape) # in eval , 3 480 854 ( channel height width )
    #x -> 1 3 w h 
    #x=torch.transpose(x,2,3)###
    x = (x - x.min()) / (x.max() - x.min())
    a = torch.cat((x[0].unsqueeze(0).cpu(), prediction.unsqueeze(0), y_visu.unsqueeze(0)), dim=0)

    return make_grid(a, normalize=normalize)



def draw(uncertainty, args):
    """ Draw a viridis map of the uncertainty
        uncertainty -> Tensor: the uncertainty map
        args        -> Argparse: global arguments
    return:
        images  -> Tensor: the viridis map """
    w, h = args.size
    yellow = torch.FloatTensor([1, 1, 0]).to(args.device).view(1, 3, 1, 1).expand(1, 3, w, h)
    blue = torch.FloatTensor([0, 0, .4]).to(args.device).view(1, 3, 1, 1).expand(1, 3, w, h)
    uncertainty = uncertainty.view(w, h)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    uncertainty.to(args.device)
    uncertainty = uncertainty * yellow + (1 - uncertainty) * blue
    return uncertainty
