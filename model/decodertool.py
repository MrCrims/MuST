import torch
import torch.nn as nn
import torchvision
from noise_layers.Merge_Image import center_of_mass
import kornia as K

def del_tensor(tensor,index):
    t1 = tensor[0:index]
    t2 = tensor[index+1:]
    return torch.cat([t1,t2],dim=0)

def cut_mask( pred_mask,threshold):

    labels = K.contrib.connected_components(pred_mask)
    l_ids = torch.unique(labels,return_counts=True)
    pos = l_ids[1]
    pos = torch.argwhere(pos>500)
    ids = []
    for i in pos[1:]:
        ids.append(l_ids[0][i[0]])
    H,W = labels.shape[2:]
    mask = torch.zeros((1,1,H,W))
    masks = []
    for i in ids:
        mask = (labels == i)
        masks.append(mask)
    centers = []
    main_ids = []
    final_masks = []
    for i in range(len(masks)):
        center = center_of_mass(masks[i])
        if i==0:
            main_center = center
            main_i = i
            main_ids.append(i)
        elif i!=0 and torch.sum(abs(center-main_center))<threshold:
            masks[main_i] += masks[i]
        else:
            main_center = center
            main_i = i
            main_ids.append(i)

    for i in main_ids:
        final_masks.append(masks[i])
    masks = final_masks
    for i,m in enumerate(masks):
        centers.append((center_of_mass(m),i))
    centers.sort(key = lambda x:(x[0][0],x[0][1]))
    labels = []
    for c in centers:
        labels.append(c[1])
    final_masks = []
    for m in masks:
        center = center_of_mass(m)
        pos = cut_pos(center,220)
        cut_mask = pred_mask[:,:,pos[0]:pos[1],pos[2]:pos[3]]
        pad_mask = torch.nn.functional.pad(cut_mask,(pos[2],W-pos[3],pos[0],H-pos[1]),value=0)
        final_masks.append(pad_mask)
    del(masks)
    if len(final_masks)!=0:
        c_final_mask = final_masks
        c_final_mask = torch.cat(c_final_mask,dim=1)
        c_final_mask.squeeze_(0)
        mask_ids = torch.unique(c_final_mask)[1:]
        masks = c_final_mask == mask_ids[:,None,None]
   
        if masks is None or masks.numel() == 0:
            return final_masks,labels,None
        boxes = torchvision.ops.masks_to_boxes(masks)

    else:
        return final_masks,labels,None
    return final_masks,labels,boxes.int()


def cut_pos(center,cut_size = 256):

    d = round(cut_size/2)
    if center[0] <= d and center[1] >= d:
        return [0,2 * center[0],center[1]-center[0],min(center[1]+center[0],1000)]
    elif center[0] >= d and center[1] <= d:
        return [center[0]-center[1],min(center[0]+center[1],1000),0,2 * center[1]]
    elif center[0] <= d and center[1] <= d:
        return [0,cut_size,0,cut_size]
    else:
        return [center[0]-d,min(center[0]+d,1000),center[1]-d,min(center[1]+d,1000)]

def pad_crop(image , threshold):
    h,w = image.shape[2:]
    crop = torchvision.transforms.CenterCrop(threshold)
    img = image

    if h >= threshold and w >= threshold:
        img = crop(image)
    elif h > threshold and w < threshold:
        d_y = (h - w)//2
        img = torch.nn.functional.pad(image,(0,0,d_y,d_y+1))
        img = crop(image) 
    elif h < threshold and w > threshold:
        d_x = (w - h)//2
        img = torch.nn.functional.pad(image,(d_x,d_x+1,0,0))
        img = crop(image) 
    else:
        a = threshold - h
        b = threshold - w
        d_x1 = (a)//2 if (a)%2==0 else a//2 + 1
        d_x2 = (a)//2 
        d_y1 = (b)//2 if (b)%2==0 else (b)//2 + 1
        d_y2 = (b)//2 
        img = torch.nn.functional.pad(image,(d_y1,d_y2,d_x1,d_x2)) 

    return img