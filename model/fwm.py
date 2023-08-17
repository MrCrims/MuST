import numpy as np
import torch
import torch.nn as nn
import os

from options import FWM_Options
# from model.encoder import Encoder
# from model.decoder import Decoder
from model.encoder_decoder import Encoder_Decoder
from model.discriminator import Discriminator
from unet import Unet
from noise_layers.Merge_Image import Image_Merge
# import lpips
import random

class FWM(nn.Module):
    def __init__(self, config : FWM_Options , device : torch.device , this_run_folder:str , writer) -> None:
        super(FWM,self).__init__()
     
        
        self.localizer = Unet(3,1).to(device).eval()
        self.discriminator = Discriminator(config).to(device)

       
        self.optimizer_loc = torch.optim.AdamW(self.localizer.parameters())
        self.optimizer_dis = torch.optim.AdamW(self.discriminator.parameters())
        contrast_factor = (0.8,1.2)
        brightness_factor = (-0.2,0.2)
        image_merge = Image_Merge(device , config.resize_bound,contrast_factor,brightness_factor,(0.1,1),(7,7))
        self.enc_dec = Encoder_Decoder(config,image_merge,self.localizer).to(device)
        self.optimizer_enc_dec = torch.optim.AdamW(self.enc_dec.parameters(),lr=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_enc_dec,[350,600],0.1)
        self.bce_loss = nn.BCELoss().to(device)
        self.l1_loss = nn.SmoothL1Loss().to(device)
        # self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        self.config = config
        self.device = device
        self.this_run_folder = this_run_folder

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.cover_label = 1
        self.encoded_label = 0

        if writer is not None:
            self.writer = writer

    def train_on_batch(self , batch:list , save_id:int):
        bg , images , masks , messages = batch
        batch_size = images.shape[0]

     
        self.enc_dec.train()
      
        self.discriminator.train()

        with torch.enable_grad():
            # Train discriminator
            input_enc,im_w,decoded_messages , messages,final_mask,merge_image,encoded_images,pred_mask = self.enc_dec(images,masks,messages,bg,True,no_noise=True)
            self.optimizer_dis.zero_grad()

            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            

          

            d_on_cover = self.discriminator(input_enc)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()
            d_on_encoded = self.discriminator(im_w.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            d_loss_on_encoded.backward()
            self.optimizer_dis.step()

      
            self.optimizer_enc_dec.zero_grad()
 
            g_loss_enc = self.mse_loss(input_enc,im_w)

            if not torch.is_tensor(decoded_messages):
                return None,(None,None,None)
            d_on_encoded_for_enc = self.discriminator(im_w)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())
            
 
            g_loss_dec = self.mse_loss(decoded_messages, messages)
   
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                    + self.config.decoder_loss * g_loss_dec
            g_loss.backward()

            self.optimizer_enc_dec.step()            


        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
            decoded_messages.shape[0] * messages.shape[1])
        

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'lr_dec         ': self.optimizer_enc_dec.param_groups[0]['lr']
        }
        return losses, (merge_image, pred_mask, final_mask)
    


    def train_on_batch_loc(self , batch:list , save_id:int):
        bg , images , masks , messages = batch
        batch_size = images.shape[0]

       
        self.localizer.train()

        with torch.enable_grad():
            # Train discriminator

            # train encoder decoder localizer
            self.optimizer_loc.zero_grad()
           
            merge_image , final_mask , labels = self.image_merge(bg , images , masks)
            pred_mask = self.localizer(merge_image)
            pred_mask = torch.nn.functional.sigmoid(pred_mask)
           
            

            g_loss_loc = self.bce_loss(pred_mask.float(),final_mask.float())
            g_loss =   self.config.localization_loss * g_loss_loc

            g_loss.backward()
            self.optimizer_loc.step()

        
        losses = {
            'loss           ': g_loss.item(),
            'loc_loss       ': g_loss_loc.item(),
        }
        return losses, (merge_image,pred_mask,final_mask)

    def validation_on_batch_loc(self , batch:list):
        bg , images , masks , messages = batch
        batch_size = images.shape[0]

        self.localizer.eval()

        with torch.no_grad():
            # Train discriminator

            

            # train encoder decoder localizer
            
            
            merge_image , final_mask , labels = self.image_merge(bg , images , masks)
            pred_mask = self.localizer(merge_image)

            pred_mask = torch.nn.functional.sigmoid(pred_mask)
           
            

            g_loss_loc = self.bce_loss(pred_mask.float(),final_mask.float())
            g_loss =   self.config.localization_loss * g_loss_loc


            


        losses = {
            'loss           ': g_loss.item(),
            'loc_loss       ': g_loss_loc.item(),
        }
        return losses, ( merge_image)

    def validation_on_batch(self , batch:list):
        bg , images , masks , messages = batch
        batch_size = images.shape[0]
        self.enc_dec.eval()

        self.discriminator.eval()

        with torch.no_grad():
            # Train discriminator
            

            input_enc,im_w,decoded_messages , messages,final_mask,merge_image,encoded_images,pred_mask = self.enc_dec(images,masks,messages,bg,False,no_noise=True)

            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            



            d_on_cover = self.discriminator(input_enc)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_on_encoded = self.discriminator(im_w.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            g_loss_enc = self.mse_loss(input_enc,im_w)

            if not torch.is_tensor(decoded_messages):
                return None,(None,None,None)
            d_on_encoded_for_enc = self.discriminator(im_w)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())
            

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                    + self.config.decoder_loss * g_loss_dec
  
            

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
            batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (merge_image, pred_mask, final_mask)

    def to_stirng(self):
        return '{}\n{}\n{}'.format(str(self.enc_dec),str(self.localizer),str(self.discriminator))


