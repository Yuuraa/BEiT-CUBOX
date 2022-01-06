import mmcv
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms
import tqdm


def show_result(
                dataset,
                img,
                result,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                opacity=0.5):
    """Draw `result` over `img`.
    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor): The semantic segmentation results to draw over
            `img`.
        palette (list[list[int]]] | np.ndarray | None): The palette of
            segmentation map. If None is given, random palette will be
            generated. Default: None
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
    seg = result[0]
    if palette is None:
        if dataset.PALETTE is None:
            print("no_palette")
            palette = np.random.randint(
                0, 255, size=(len(dataset.CLASSES), 3))
        else:
            palette = dataset.PALETTE
    palette = np.array(palette)
    assert palette.shape[0] == len(dataset.CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    assert 0 < opacity <= 1.0
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg -1 == label, :] = color # GT 이미지는 라벨이 1 ~ 200의 값으로 저장되었기 때문에 -1을 해줘야 올바른 palette가 매핑 되게 됨
    # convert to BGR
    color_seg = color_seg[..., ::-1]
    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    # if not (show or out_file):
    #     warnings.warn('show==False and out_file is not specified, only '
    #                   'result image will be returned')
        return img


def viz_gt(dataset, img_path, gt_path, split, save_path):
    img_path = f"{img_path}/{split}"
    gt_path = f"{gt_path}/{split}"
    print(img_path, gt_path)

    classes = set()
    # for occ in ["none", "wireloose", "wiremedium", "wiredense", "semitransparent"]:
        # for img_path in tqdm.tqdm(glob.glob(f"{img_path}/{occ}/**.**")): # TODO: 데이터셋 뭐 사용했냐에 따라 다름.. .지금 보니까 도커 이미지에 cubox_dataset/original의 것이 아니라 다른 폴더가 마운트 되어서, wiremedium/none 등등의 폴더가 분리 되어 있지 않음 (로컬에서의 문제)
    for img_file in tqdm.tqdm(sorted(glob.glob(f"{img_path}/**.**"))):
        img_name = img_file.split("/")[-1].split(".")[0]
        # gt_path = f"{gt_path}/{occ}/{img_name}.png"
        gt_file = f"{gt_path}/{img_name}.png"
        # print(gt_file)
        gt_img = Image.open(gt_file)
        gt_tensor = transforms.ToTensor()(gt_img) * 255
        classes.add(int(gt_tensor.max().item()))
        # print(gt_tensor.max(), gt_tensor.min(), gt_tensor.unique())
        show_result(dataset, img_file, gt_tensor, show=True, out_file=f"{save_path}/{img_name}.png")
    print(len(classes))
    print(classes)

if __name__ == "__main__":
    from cubox_dataset import CUBOXDataset
    import os

    dataset = CUBOXDataset
    # img_path = "/mnt/disk1/cubox_dataset/original/images"
    # gt_path = "/mnt/disk1/cubox_dataset/original/seg_map"
    img_path = "/dataset/images"
    gt_path = "/dataset/seg_map"
    # save_path = "/home/yura/Computer_Vision_LAB/Semantic_Segmentation/BEiT-CUBOX/visualization/gt"
    save_path = "/unilm/visualization/gt"
    os.makedirs(save_path, exist_ok=True)
    viz_gt(dataset, img_path, gt_path, "test", save_path)