import torch
import torch.nn as nn
from options import FWM_Options
from model.conv_bn_relu import ConvBNRelu_LN,ConvBNRelu,Resblock,SEblock,SE_Resblock,Image_SEblock
import torchvision

class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: FWM_Options):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]
       
        for i in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)
            if i % 2==0 and i>0:
                layers.append(SEblock(self.conv_channels,8))
        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels+ config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels+3, 3, kernel_size=1)

    def forward(self, image, mask ,message):
        H,W = image.shape[2:]
        c_masks = mask.permute(1,0,2,3)
       
        c_masks.squeeze_(0)
        mask_ids = torch.unique(c_masks)[1:]
        masks = c_masks == mask_ids[:,None,None]
        print(masks.shape)
        boxes = torchvision.ops.masks_to_boxes(masks)
        boxes = boxes.int()
        cut_imgs = image * mask
        s_cut_imgs = torch.split(cut_imgs,1)
        s_cut_masks = torch.split(mask,1)
        origin_shape = []
        out_rec = []
        out_mask = []
        print(boxes)
        if len(boxes)==0:
            boxes = [[0,0,H,W]*image.shape[0]]
        for i in range(len(s_cut_imgs)):
            x1,y1,x2,y2 = boxes[i]
            rectangle = s_cut_imgs[i][:,:,y1:y2,x1:x2]
            rec_mask = s_cut_masks[i][:,:,y1:y2,x1:x2]
            origin_shape.append((rectangle.shape[2],rectangle.shape[3]))
            if rectangle.shape[2]!=256 or rectangle.shape[3] != 256:
                    rectangle = torch.nn.functional.interpolate(rectangle,size=(256,256))
                    rec_mask = torch.nn.functional.interpolate(rec_mask,size=(256,256))
            out_rec.append(rectangle)
            out_mask.append(rec_mask)
        
        input_enc = torch.cat(out_rec,dim=0)


        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, 256, 256)
        encoded_image = self.conv_layers(input_enc)
        
        concat = torch.cat([expanded_message, encoded_image], dim=1)
        im_w = self.after_concat_layer(concat)
   
        concat = torch.cat([im_w,input_enc], dim=1)
        im_w = self.final_layer(concat)
        s_img = torch.split(im_w,1)
        mask_out = []
        enc_image = []
        for i in range(len(s_img)):
            x1,y1,x2,y2 = boxes[i]
            a = torch.nn.functional.interpolate(s_img[i],size=origin_shape[i])
            a = torch.nn.functional.pad(a,(x1,W-x2,y1,H-y2),value=0)
            enc_image.append(a)
            a = torch.nn.functional.interpolate(out_mask[i],size=origin_shape[i])
            a = torch.nn.functional.pad(a,(x1,W-x2,y1,H-y2),value=0)
            mask_out.append(a)
        mask = torch.cat(mask_out,dim=0)
        encoded_images = torch.cat(enc_image,dim=0)
        encoded_images = encoded_images * mask

        s_img = torch.split(input_enc,1)
        ori_img = []
        for i in range(len(s_img)):
           x1,y1,x2,y2 = boxes[i]
           a = torch.nn.functional.interpolate(s_img[i],size=origin_shape[i])
           a = torch.nn.functional.pad(a,(x1,W-x2,y1,H-y2),value=0)
           ori_img.append(a)
        ori_img = torch.cat(ori_img,dim=0)
        ori_img = ori_img * mask
        return encoded_images,input_enc,im_w,ori_img,origin_shape,mask