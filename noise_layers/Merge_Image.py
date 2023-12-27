import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import kornia as K

def CheckPositions(check , positions):
    for i in positions:
        if (check[0]>=(i[0]+i[2]+20) or check[0]+check[2] <=i[0]-20) or (check[1]>=(i[1]+i[3]+20) or check[1]+check[3] <=i[1]-20):
                continue
        else:
            return False
    return True

def center_of_mass(image:torch.Tensor):
    center = torch.argwhere(image).float().mean(0)
    center = torch.round(center[2:])
    return center.int()


def Img_Reszie_Merge(device,background , images , masks , gaussianblur , brightness_factor , contrast_factor , resize_rate : list, istrain):
    # first arrange location
    # second stamp on image
    # gaussain blur
    
    g_images = gaussianblur(images)
    # remove edge
    if not istrain:
        r_masks = F.interpolate(masks,size=(620,620))
        r_masks = F.pad(r_masks,(10,10,10,10),value=0)
        r_images = r_masks*g_images
        # Feather
        d_masks = F.interpolate(masks,size=(630,630))
        d_masks = F.pad(d_masks,(5,5,5,5),value=0)
        d_masks = d_masks-r_masks
        d_images = r_images * d_masks
        gaussianblur = torchvision.transforms.GaussianBlur((7,7),sigma=2)
        g_d_images = gaussianblur(d_images)
        images = (1 - d_masks)*r_images + g_d_images
    else :
        images = g_images
    n = images.shape[0]
    images = torch.split(images , 1 , dim=0)
    if not istrain:
        masks = torch.split(r_masks , 1 ,dim=0)
    else:
        masks = torch.split(masks,1,dim=0)
    wb , hb = background.shape[2:]
    imgs = []
    postions = []
    for i in range(0,n):
        flag = 0
        img = F.interpolate(images[i] , scale_factor=(np.sqrt(resize_rate[i]),np.sqrt(resize_rate[i])) , mode='nearest')
        imgs.append(img)
        w_i , h_i = img.shape[2:]
        y = np.random.randint(0,hb-h_i)
        x = np.random.randint(0,wb-w_i)
        while not flag:
            if len(postions) == 0:
                break
            while not CheckPositions([x,y,w_i,h_i],postions):
                y = np.random.randint(0,hb-h_i)
                x = np.random.randint(0,wb-w_i)
            flag = 1
        postions.append([x,y,w_i,h_i])
    final_masks = torch.zeros((1,1,wb,hb)).to(device)
    pos = []
    merge_img = background
    cut_imgs = []
    for i in range(0,n):
        mask = F.interpolate(masks[i] , scale_factor=(np.sqrt(resize_rate[i]),np.sqrt(resize_rate[i])) , mode='nearest')
        cut_img = mask * imgs[i]
        # contrast and brightness
        cut_img = K.enhance.adjust_brightness(cut_img,brightness_factor[i],clip_output=False)
        cut_img = K.enhance.adjust_contrast(cut_img,contrast_factor[i],clip_output=False)
        cut_img = mask * cut_img
        postion = postions[i]  
        pad_mask = F.pad(mask , (postion[0] , wb-postion[0]-postion[2] , postion[1] , hb-postion[1]-postion[3]),value=0)
        pad_cut_img = F.pad(cut_img , (postion[0] , wb-postion[0]-postion[2] , postion[1] , hb-postion[1]-postion[3]),value=0)
        merge_img = merge_img * (1 - pad_mask) + pad_cut_img
        final_masks += pad_mask  # prepare for mask segmentation
        center = center_of_mass(pad_mask)
        pos.append((center,i))
    pos.sort(key = lambda x:(x[0][0],x[0][1]))
    labels_pos = {}
    for i in range((len(pos))):
        labels_pos[i] = pos[i]
    return merge_img , final_masks , labels_pos

class Image_Merge(nn.Module):
    def __init__(self , device , resize_bound:list , contrast_factor , brightness_factor, sigma , kernel_size) -> None:
        super().__init__()
        self.device = device
        self.resize_bound = resize_bound
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.gaussianblur = torchvision.transforms.GaussianBlur(kernel_size,sigma)

    def forward(self , background , images , masks ,istrain=False):
        resize_rate = [ random.uniform(self.resize_bound[0] , self.resize_bound[1]) for i in range(0,len(images)) ]
        contrast_factor = [random.uniform(self.contrast_factor[0],self.contrast_factor[1]) for i in range(0,len(images))]
        brightness_factor = [random.uniform(self.brightness_factor[0],self.brightness_factor[1]) for i in range(0,len(images))]
        
        merge_img , final_mask , label_pos = Img_Reszie_Merge(self.device,background , images , masks , self.gaussianblur , brightness_factor , contrast_factor ,resize_rate,istrain)
        return merge_img , final_mask , label_pos 