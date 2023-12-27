import numpy as np
import torch
import torch.nn as nn
import torchvision
from options import FWM_Options
from model.conv_bn_relu import ConvBNRelu,Resblock_LN,Resblock,SEblock,SE_Resblock
from noise_layers.Merge_Image import center_of_mass
import kornia as K
import time
from model.decodertool import *

class Decoder(nn.Module):
    """
    
    """
    def __init__(self, config: FWM_Options):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels
         
        layers = [Resblock(3, self.channels)]
        layers.append(SEblock(self.channels,8))

        for i in range(config.decoder_blocks - 1):
            layers.append(Resblock(self.channels, self.channels))
            if i % 2==0 and i>0:
                layers.append(SEblock(self.channels,8))
        layers.append(Resblock(self.channels, config.message_length))
        layers.append(SEblock(config.message_length,1))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        
        self.threshold = 256
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, messages ,pred_mask, merge_img,no_noise=False):
        if no_noise:
            input_imgs = merge_img
        else:
            pred_mask[pred_mask>0] = 1
            pred_mask[pred_mask<1] = 0
            masks, labels,boxes = cut_mask(pred_mask,self.threshold)
            if boxes is None:
                return [False,False]
            input_imgs = [0] * len(messages)
            res = [0] * len(messages)
            for i in range(len(masks)):
                cut_img = masks[i] * merge_img
                x1,y1,x2,y2 = boxes[i]
                input_img = cut_img[:,:,y1:y2,x1:x2]
                if input_img.shape[2]!=256 or input_img.shape[3] != 256:
                    input_img = torch.nn.functional.interpolate(input_img,size=(256,256))
                input_imgs[labels[i]] = input_img
            id = []
            for i in range(len(messages)):
                if not torch.is_tensor(input_imgs[i]):
                    # print(1)
                    id.append(i)
            step = 0
            for i in id:
                index = i if step == 0 else i - step
                input_imgs.pop(index)
                step += 1
            
            input_imgs = torch.cat(input_imgs,dim=0)
        x = self.layers(input_imgs)
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        l = len(messages)

        if not no_noise:
            for i in range(l):
                if labels.count(i)==0:
                    messages = del_tensor(messages,i)

        
        return x,messages