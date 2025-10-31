import cv2
import numpy as np
import os
from PIL import Image
# reading the damaged image
damaged_img = cv2.imread(filename=r'/home/haotian/PAD/result/run-PAD-CD/8.png')
input_path = "/home/haotian/rosbag2image/bagfile/result/"
save_path = "/home/haotian/rosbag2image/bagfile/Mask_Generation/"
data_dir = input_path
data_files = os.listdir(data_dir)
# get the shape of the image
height, width = damaged_img.shape[0], damaged_img.shape[1]
for data_file in data_files:
    print(data_file)
    name = data_file.split(".")[0]
    impath = data_dir + data_file
    print(impath) 
    ori_img = cv2.imread(filename=impath)
    ori_height, ori_width = ori_img.shape[0], ori_img.shape[1] 
    #ori_img = Image.open(impath).convert('RGB')
    #ori_width, ori_height = ori_img.size
    print("ori_height , ori_width", ori_height, ori_width)
    for i in range(ori_height):
        for j in range(ori_width):
            if ori_img[i, j].sum() > 0:#damaged_img[i, j].sum() > 0:
                ori_img[i, j] = 0#damaged_img[i, j] = 0
            else:
                ori_img[i, j] = [255, 255, 255]#damaged_img[i, j] = [255, 255, 255]
            # saving the mask 
    mask = ori_img
            #cv2.imwrite('mask.png', mask)
    cv2.imwrite(save_path+name+"_mask001"+".png",mask)
'''
# Converting all pixels greater than zero to black while black becomes white
for i in range(height):
    for j in range(width):
        if damaged_img[i, j].sum() > 0:
            damaged_img[i, j] = 0
        else:
            damaged_img[i, j] = [255, 255, 255]
''' 
# displaying mask
#cv2.imshow("damaged image mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
