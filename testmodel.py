import os
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision
import utils.dataloader as dataloader
import utils.utils as utils
from model.fwm import *
from model.decoder import cut_mask
from kornia.metrics import psnr,ssim,mean_iou
import torch.nn.functional as F
from torchvision import transforms

def save_img(origin_img,filename):
    img = origin_img[:,:,:,:].cpu()
    img = (img + 1) / 2
    torchvision.utils.save_image(img,filename,normalize=False)

def IoU(pred,gt):
    t_and = torch.bitwise_and(pred,gt)
    t_or = torch.bitwise_or(pred,gt)
    return t_and.float().mean()/t_or.float().mean()


def test_enc_dec(model:FWM,input_num,scale_rate,contrast_factor,brightness_factor,device):
    bg_loader = dataloader.dataloader_background("/public/wangguanjie/data/Pure_Commercial_Image/",1)# default batchsize of  background 1
    val_loader = dataloader.dataloader_img_mask("/public/wangguanjie/data/Pure_Commercial_Image/val",input_num)
    contrast_factor =contrast_factor
    brightness_factor = brightness_factor
    image_merge = Image_Merge(device , scale_rate,contrast_factor,brightness_factor,(0.1,1),(7,7))
    total_rate = 0.
    total_size = 0
    total_psnr = 0.
    total_ssim = 0.
    step = 0
    
    trans = transforms.ToTensor()
    with torch.no_grad():
        for ims,masks in val_loader:
            ims = ims.to(device)
            masks = masks.to(device)
            
            masks[masks>0]=1
            masks[masks<1]=0
            save_img(masks,"save_images/mask{}.png".format(step))
            input_i = ims * masks
            if ims.shape[0] != input_num:
                return total_rate/total_size
            messages = torch.Tensor(np.random.choice([0, 1], (ims.shape[0], 30))).to(device)
           
            background = bg_loader.__iter__().__next__()
            background = background.to(device)
            encode_images,input_enc,im_w,ori_img = model.enc_dec.encoder(input_i,masks,messages)
          

            sace_enc = encode_images + (1-masks)*ims
            merge_image , final_mask , label_pos = image_merge(background,encode_images,masks,False)
            
            pred_mask = model.localizer(merge_image)
            
            decoded_messages,messages = model.enc_dec.decoder(messages , label_pos, pred_mask , merge_image)
            
            if not torch.is_tensor(decoded_messages):
                continue 
            decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
           
            bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                decoded_messages.shape[0] * messages.shape[1])
        
            total_rate += bitwise_avg_err
            total_size += 1
            total_psnr += psnr(((input_enc+1)/2).clamp(0,1),((im_w+1)/2).clamp(0,1),1.).item()
            total_ssim += ssim(((ims+1)/2).clamp(0,1),((sace_enc+1)/2).clamp(0,1),7).mean().item()
            # tobe_save = torch.cat([ims,sace_enc,sace_enc-ims],dim=0)
            # print("Average psnr:",total_psnr/len(val_loader),"SSIM:",total_ssim/len(val_loader))
            save_img(sace_enc,"save_images/{}.png".format(step))
            save_img(input_enc,"save_images/input{}.png".format(step))
            save_img(im_w,"save_images/output{}.png".format(step))
            save_img(merge_image*final_mask,"save_images/cut{}.png".format(step))
            save_img(final_mask,"save_images/finalmask{}.png".format(step))
            save_img(merge_image,"save_images/merge_{}.png".format(step))
            
            
            
            step += 1

    
    
    return total_rate/total_size


def test_loc(model, input_num, scale_rate,device ):
    bg_loader = dataloader.dataloader_background("/public/wangguanjie/data/Pure_Commercial_Image/",1)# default batchsize of  background 1
    val_loader = dataloader.dataloader_img_mask("/public/wangguanjie/data/Pure_Commercial_Image/val",input_num)
    image_merge = Image_Merge(device, scale_rate)
    total_rate = 0.
    total_size = 0
    total_iou = 0.
    with torch.no_grad():
        for ims , masks in val_loader:
            ims = ims.to(device)
            masks = masks.to(device)
            masks[masks>0]=1
            masks[masks<1]=0
            # print(ims.shape,masks.shape)
            if ims.shape[0] != input_num:
                return total_rate/total_size
            
            background = bg_loader.__iter__().__next__()
            background = background.to(device)
            merge_image , final_mask , label_pos = image_merge(background,ims,masks)
            pred_mask = model(merge_image)
            pred_mask[pred_mask>0] = 1
            pred_mask[pred_mask<1] = 0
            c_masks, labels, boxes = cut_mask(label_pos,pred_mask)
            total_rate += (len(c_masks))/ims.shape[0]
            total_iou += IoU(pred_mask.int(),final_mask.int()).item()
            total_size += 1
            print(total_iou/total_size)
    return total_rate/total_size

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


if __name__=="__main__":
    setup_seed(2023)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print((device))

    with open("","rb") as f:
 
        train_options = pickle.load(f)
        fwm_options = pickle.load(f)

    checkpoint = torch.load("")

    fwm = FWM(fwm_options,device,"./runs",None)
    fwm.enc_dec.encoder.load_state_dict(checkpoint['enc-model'],strict = False)
    fwm.enc_dec.decoder.load_state_dict(checkpoint['dec-model'],strict = False)
    fwm.localizer.load_state_dict(checkpoint['loc-model'],strict = False)
    
    fwm.enc_dec.encoder.eval()
    fwm.localizer.eval()
    fwm.enc_dec.decoder.eval()
    # acc = test_loc(fwm.localizer,3,[0.16,0.16],device)
    # print(acc)
    # brightness_factors = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
    # for brightness_factor in brightness_factors:
    err = test_enc_dec(fwm,1,[0.16,0.16],(1.0,1.0),(0.0,0.0),device)
    print("BRIGHTNESS FACTOR={}".format(0.0))
    print("ACC: ",(1-err)*100)
