import PIL
from PIL import Image
from PIL import ImageDraw
import torch
import torchvision
from torchvision.transforms import ToPILImage
import os
import seaborn as sns
import matplotlib.pyplot as plt


# SAVE_DIR = f"{os.path.realpath(__file__)}/test_attn_map"
SAVE_DIR = "/unilm/beit/semantic_segmentation/test_attn_map"
TENSOR_SAVE_DIR = "/unilm/beit/semantic_segmentation/test_attn_tensor"
os.makedirs(SAVE_DIR, exist_ok=True)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        new_tensor = torch.zeros_like(tensor)
        # RGB
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            new_tensor[i] = (t * s) + m
        return new_tensor


def recover_img(denormalizer: Denormalize, to_pil: torchvision.transforms, img_tensor: torch.Tensor) -> PIL.Image:
    denorm_img_tensor = denormalizer(img_tensor)
    denorm_img = to_pil(denorm_img_tensor)
    denorm_img = PIL.ImageOps.invert(denorm_img)

    return denorm_img


def annot_img(img_pil, point1, point2):
    draw = ImageDraw.draw(img_pil)
    draw.rectangle((point1, point2), outline=(255, 0, 0))
    return img_pil


def save_heatmap_attn_imgs(attn_tensor_onehead, L, save_dir):
    for i in range(1, L):
        heatmap = sns.heatmap(attn_tensor_onehead[:, i][1:].reshape((16, 16)))
        heatmap_figure = heatmap.get_figure()
        heatmap_figure.savefig(f'{save_dir}/attn_{i}.png', dpi=400)
        plt.clf()


def save_img(img_tensor, attn_tensor, ori_filename, h_idx, w_idx):
    denormalizer = Denormalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])
    to_pil = ToPILImage()
    recovered_img = recover_img(denormalizer, to_pil, img_tensor)
    new_save_dir = f"{SAVE_DIR}/{ori_filename}_{h_idx}_{w_idx}"
    os.makedirs(new_save_dir, exist_ok=True)
    recovered_img.save(f"{new_save_dir}/cropped_img.jpg")
    
    # save_tensor(img_tensor, attn_tensor, ori_filename, h_idx, w_idx)
    Heads, L, L = attn_tensor.shape
    for h in range(Heads):
        new_attn_dir = f"{new_save_dir}/head_{h}"
        os.makedirs(new_attn_dir, exist_ok=True)
        save_heatmap_attn_imgs(attn_tensor[h], L, new_attn_dir)


def save_tensor(img_tensor, attn_tensor, ori_filename, h_idx, w_idx):
    new_save_dir = f"{TENSOR_SAVE_DIR}/{ori_filename}_{h_idx}_{w_idx}"
    os.makedirs(new_save_dir, exist_ok=True)

    torch.save(img_tensor, f"{new_save_dir}/cropped_img.pt")
    torch.save(attn_tensor, f"{new_save_dir}/attention.pt")
    