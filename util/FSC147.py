import numpy as np
import torch.nn.functional as F
import math
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF

MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

class resizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample

class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample

class resizePreTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is preserved
    Density and boxes correctness not preserved(crop and horizontal flip)
    """
    
    def __init__(self, MAX_HW=384):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        '''scale_factor = float(256)/ H
        new_H = 16*int(H*scale_factor/16)
        new_W = 16*int(W*scale_factor/16)'''
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)

        if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = PreTrainNormalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample

class resizeTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image, while boxes are not preserved to be in the cropped area.
    """
    
    def __init__(self, MAX_HW=384):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)

        if new_count > 0: resized_density = resized_density * 100 * (orig_count / new_count)

        # Random 384*384 crop in a new_W*384 image and 384*new_W density map
        start = random.randint(0, new_W-1-383)

        reresized_image = TF.crop(resized_image, 0, start, 384, 384)
        reresized_density = resized_density[:, start:start+384]
            
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        reresized_image = Normalize(reresized_image)
        reresized_density = np.transpose(reresized_density)
        reresized_density = torch.from_numpy(reresized_density)       
        sample = {'image':reresized_image,'boxes':boxes,'gt_density':reresized_density}
        return sample

PreTrainNormalize = transforms.Compose([   
        transforms.RandomResizedCrop(384, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

Normalize = transforms.Compose([   
        transforms.ToTensor(),
        #transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

Transform = transforms.Compose([resizeImage( MAX_HW)])
TransformTrain = transforms.Compose([resizeTrainImage(MAX_HW)])
TransformPreTrain = transforms.Compose([resizePreTrainImage(MAX_HW)])