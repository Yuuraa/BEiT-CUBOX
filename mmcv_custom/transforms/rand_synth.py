from mmseg.datasets.builder import PIPELINES
import torchvision.transforms as transforms
import torch
from PIL import Image, ImageChops
import numpy as np


@PIPELINES.register_module()
class RandomSynthAug(object):
    """
    Random Augmentation with Artificial Wires
    """
    def __init__(self, img_size, degree=60):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=degree, shear=60),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.CenterCrop(img_size)
        ])
        self.patterns = []
        self.p = 0.2
        for density in [9, 10, 11, 12, 13, 16, 17, 100]:
            tmp = Image.open('/unilm/mmcv_custom/transforms/patterns/{}x.jpg'.format(density))
            self.patterns.append(np.array(tmp))
        tmp.close()

    def __call__(self, results):
        if self.p < torch.rand(1):
            return results
        else:
            pattern_id = np.random.choice(len(self.patterns))
            pattern = self.transform(self.patterns[pattern_id])
            comb = ImageChops.multiply(Image.fromarray(results['img']), pattern)
            results['img'] = np.array(comb)

            return results

