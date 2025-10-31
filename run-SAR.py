import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import os

import math
from PIL import Image

from heatmap_MI import img_heatmap_mi
from heatmap_CD import img_heatmap_cd
from fuse_filter import fuse_heatmap, heatmap_filter

iou_thre = 0.5
ratio_mi = 0.5 # ratio_cd = 1-ratio_mi
kernel_pram = 80
thresh_pram = 80 # percentile, from small to big
input_path = "/home/haotian/FastSAM_MI_GAN/test_img/"
save_path = "/home/haotian/FastSAM_MI_GAN/result/run-PAD-CD/e61v2/"
save_path2 = "/home/haotian/FastSAM_MI_GAN/result/run-PAD-CD/e61v2/"
#input_path = "/home/haotian/rosbag2image/bagfile/image_0.4/"
#save_path = "/home/haotian/rosbag2image/bagfile/Patch_Remove_Result_0.4/"
#save_path2 = "/home/haotian/rosbag2image/bagfile/Patch_Localization_Map_Mask_0.4/"
#final_map_path = "/home/dell/jlh/my_patch_defense/code/inria_P3_final_map"

import argparse
import warnings
from pathlib import Path

import pickle
import PIL.Image
import time
from tqdm import tqdm

from lib.model_zoo.migan_inference import Generator as MIGAN
from lib.model_zoo.comodgan import (
    Generator as CoModGANGenerator,
    Mapping as CoModGANMapping,
    Encoder as CoModGANEncoder,
    Synthesis as CoModGANSynthesis
)
from fastsam import FastSAM
def get_mask(image, mask_generator):
    
    masks = mask_generator.generate(image.astype(np.uint8))
    return masks

#def read_mask(mask_path, invert=False):
def read_mask(mask, invert=False):
    #mask = Image.open(mask_path)
    mask = resize(mask, max_size=512, interpolation=Image.NEAREST)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0
    return Image.fromarray(mask).convert("L")


def resize(image, max_size, interpolation=Image.BICUBIC):
    w, h = image.size
    if w > max_size or h > max_size:
        resize_ratio = max_size / w if w > h else max_size / h
        image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
    return image


def preprocess(img: Image, mask: Image, resolution: int) -> torch.Tensor:
    img = img.resize((resolution, resolution), Image.BICUBIC)
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    img = np.array(img)
    mask = np.array(mask)[:, :, np.newaxis] // 255
    img = torch.Tensor(img).float() * 2 / 255 - 1
    mask = torch.Tensor(mask).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    mask = mask.permute(2, 0, 1).unsqueeze(0)
    x = torch.cat([mask - 0.5, img * mask], dim=1)
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="One of [migan-256, migan-512, comodgan-256, comodgan-512]", required=False, default="migan-512")
    parser.add_argument("--model-path", type=str, help="Saved model path.", required=False, default="./models/migan_512_places2.pt")
    parser.add_argument("--invert-mask", action="store_false", help="Invert mask? (make 0-known, 1-hole)", default="True")
    parser.add_argument("--output-dir", type=Path, help="Output directory.", required=False, default="./run-PAD-CD/e61v2")
    parser.add_argument("--device", type=str, required=False, help="Device.", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cuda = False
    if args.device == "cuda":
        cuda = True
    

    if args.model_name == "migan-256":
        resolution = 256
        model = MIGAN(resolution=256)
    elif args.model_name == "migan-512":
        resolution = 512
        model = MIGAN(resolution=512)
    elif args.model_name == "comodgan-256":
        resolution = 256
        comodgan_mapping = CoModGANMapping(num_ws=14)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    elif args.model_name == "comodgan-512":
        resolution = 512
        comodgan_mapping = CoModGANMapping(num_ws=16)
        comodgan_encoder = CoModGANEncoder(resolution=resolution)
        comodgan_synthesis = CoModGANSynthesis(resolution=resolution)
        model = CoModGANGenerator(comodgan_mapping, comodgan_encoder, comodgan_synthesis)
    else:
        raise Exception("Unsupported model name.")
    model.load_state_dict(torch.load(args.model_path))
    if cuda:
        model = model.to("cuda")
    model.eval()
    
    device = "cuda:0"

    
    ## FastSAM
    modelFastSAM = FastSAM('./weights/FastSAM.pt')


    print(save_path)
    folder = os.path.exists(save_path)

    if not folder:
        os.makedirs(save_path)

    with torch.no_grad():
        data_dir = input_path
        data_files = os.listdir(data_dir)
        for data_file in data_files:
            print(data_file)
            name = data_file.split(".")[0]
            impath = data_dir + data_file
            
            ori_img = Image.open(impath).convert('RGB')
            img_resized = resize(ori_img, max_size=resolution)
            ori_width, ori_height = ori_img.size
            print("ori_height , ori_width", ori_height, ori_width)
            print("--invert-mask",args.invert_mask)
            
            #mi_img, cd_img, fuse_img = fuse_heatmap(impath, ori_height, ori_width)
            time_start = time.time()
            cd_img = fuse_heatmap(impath, ori_height, ori_width)
            threshold = np.percentile(cd_img, thresh_pram)
            h_t, h_t_o, h_t_o_c, h_t_o_c_o = heatmap_filter(cd_img, threshold, ori_height, ori_width)


            gray = np.where(h_t_o_c_o >0,1,0)

            rgb_color = cv2.imread(impath)

            image = cv2.cvtColor(rgb_color, cv2.COLOR_BGR2RGB)
            DEVICE = "cuda:0"
            everything_results = modelFastSAM(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
            for result in everything_results:
                mask = result.masks.data
            #just for Dpatch
            #image = cv2.resize(image,(416,416))

            h = image.shape[0]
            w = image.shape[1]
            #mask = get_mask(image, mask_generator) # getting the mask from SAM

            #check if the adversary patch got isolated or not
            #print("len(mask)",len(mask))
            #visualize the segmentation
            #plt.figure(figsize=(20,20))
            #plt.imshow(image)
            #show_anns(mask)
            #plt.axis('off')
            #plt.show() 
            result_mask = np.zeros((h,w))
            for k in range(len(mask)):

                #mask_k = mask[k].get('segmentation')
                mask_k = result.masks.data[k] == 1.0
                mask_k = mask_k.cpu().numpy()
                n = mask_k&gray
                u = mask_k#|gray
                iou = np.sum(n)/(np.sum(u))
                print("iou",iou)

                n_1 = mask_k&result_mask.astype(np.uint8)
                u_1 = mask_k
                iou1 = np.sum(n_1)/(np.sum(u_1))
                print("iou1",iou1)

                #if(iou>iou_thre and iou1<0.1):
                if(iou>0.9 and iou1<0.01):
                    mask_k_save = np.expand_dims(mask_k,axis=2)
                    mask_k_save = np.tile(mask_k_save,3)
                    rgb_color = rgb_color*(~mask_k_save) 
                    result_mask = result_mask.astype(np.uint8) | mask_k
                    '''mask_k_save = np.expand_dims(mask_k,axis=2)
                    mask_k_save = np.tile(mask_k_save,3)
                    mask_gray = np.expand_dims(mask_k*128,axis=2)
                    mask_gray = np.tile(mask_gray,3)
                    rgb_color = rgb_color*(~mask_k_save) + mask_gray
                    result_mask = result_mask.astype(np.uint8) | mask_k'''
                    '''result_mask = result_mask.astype(np.uint8) | mask_k
                    rgb_color = cv2.inpaint(rgb_color, mask_k.astype(np.uint8), 3, cv2.INPAINT_NS)'''
            result_mask = Image.fromarray(result_mask)
            mask = read_mask(result_mask, invert=args.invert_mask)
            mask_resized = resize(mask, max_size=resolution, interpolation=Image.NEAREST)
            x = preprocess(img_resized, mask_resized, resolution)
            if cuda:
                x = x.to("cuda")
            with torch.no_grad():
                result_image = model(x)[0]

            result_image = (result_image * 0.5 + 0.5).clamp(0, 1) * 255
            result_image = result_image.to(torch.uint8).permute(1, 2, 0).detach().to("cpu").numpy()

            result_image = cv2.resize(result_image, dsize=img_resized.size, interpolation=cv2.INTER_CUBIC)
            mask_resized = np.array(mask_resized)[:, :, np.newaxis] // 255
            composed_img = img_resized * mask_resized + result_image * (1 - mask_resized)
            time_cd_end = time.time()
            print('--------------cd cost %f s' %(time_cd_end-time_start))
            composed_img = Image.fromarray(composed_img)
            composed_img.save(args.output_dir / f"{Path(save_path).stem}.png")

            cv2.imwrite(save_path+name+".png",rgb_color) #save patch removed image
            #cv2.imwrite("./mask_hxx_0.05_gray/"+name+".png",cv2.inpaint(rgb_color, cv2.blur(result_mask.astype(np.uint8),(5,5)), 3, cv2.INPAINT_NS))
            #cv2.imwrite(save_path+name+".png",cv2.inpaint(rgb_color, cv2.blur(result_mask.astype(np.uint8),(5,5)), 3, cv2.INPAINT_NS))
            #cv2.imwrite(save_path2+name+"_mask001"+".png",result_mask)
