import mmcv
from mmcv.image import imdenormalize
import copy
from mmseg.datasets.builder import PIPELINES
import torchvision.transforms as transforms
from numpy import random
import torch
from PIL import Image, ImageChops
import numpy as np
import cv2


@PIPELINES.register_module()
class RandomSave(object):
    def __init__(self, keys, prob=0.5, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True):
        self.keys = keys
        self.prob = prob

        self.to_pil = transforms.ToPILImage()
        # self.mean = mean
        # self.std = std
        
        # BGR 2 RGB 
        # mean = [mean[2], mean[1], mean[0]]
        # std = [std[2], std[1], std[0]]
        # self.denorm = transforms.Normalize(mean=[-m/s for m,s in zip(mean, std)], std=[1/s for s in std])
    
    def __call__(self, results):
        data = {}
        for key in self.keys:
            data[key] = results[key]
            if random.rand() < self.prob:
                self.to_pil(self.denorm(data[key].data.detach().cpu())).save(f"/unilm/mosaic_results/{results['filename'].split('/')[-1]}")
        
        return results