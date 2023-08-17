import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from unet import Unet
from options import FWM_Options

class Encoder_Decoder(nn.Module):
    def __init__(self,config:FWM_Options,noise,localizer) -> None:
        super(Encoder_Decoder,self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.localizer = localizer
        self.noise = noise
    
    def forward(self,images,masks,messages,bg,istrain=True,no_noise=False):
        encoded_images,input_enc,im_w,ori_img,origin_shape,re_mask = self.encoder(images,masks,messages)
        if no_noise:
            decoded_messages , messages = self.decoder(messages,None,None,im_w,no_noise=no_noise)
            return input_enc,im_w,decoded_messages , messages,None,None,None,None
        else:
            merge_image , final_mask , label_pos = self.noise(bg , encoded_images , masks,istrain)
            messages = torch.split(messages,1,dim=0)
            rearrange_messages = []
            for i, m in enumerate(messages):
                rearrange_messages.append(messages[label_pos[i][1]])
            messages = torch.cat(rearrange_messages,dim=0)
            pred_mask = self.localizer(merge_image.detach())
            decoded_messages , messages = self.decoder(messages , pred_mask.detach() , merge_image,no_noise=no_noise)
            return input_enc,im_w,decoded_messages , messages,final_mask,merge_image,encoded_images,pred_mask