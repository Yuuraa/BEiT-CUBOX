
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import sys

from mmseg.apis.test import collect_results_gpu, collect_results_cpu


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.
    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test_logits(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(task_num=len(dataset), file=sys.stderr)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, return_logits=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test_logits(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    scores = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(task_num=len(dataset), file=sys.stderr)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, return_logits=True, rescale=True, **data)
            # print('result len', len(result)) # 2: pred, score
            # print('first result length', len(result[0])) # batch_size
            # print('second result length', len(result[1])) # batch_size
            # print('first result shape', result[0][0].shape) # (n_classses, w, h)
            # print('second result shape', result[1][0].shape) # (n_classes,)
        pred, score = result

        if isinstance(pred, list):
            if efficient_test:
                pred = [np2tmp(_) for _ in pred]
                score = [np2tmp(_) for _ in score]
            results.extend(pred)
            scores.extend(score)
        else:
            if efficient_test:
                pred = np2tmp(pred)
                score = np2tmp(score)
            results.append(pred)
            scores.append(score)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        # print("Collecting with GPU")
        results = collect_results_gpu(results, len(dataset))
        scores = collect_results_gpu(scores, len(dataset))
    else:
        # print("Collecgint with CPU")
        results = collect_results_cpu(results, len(dataset), 'preds_tmpdir')
        scores = collect_results_cpu(scores, len(dataset), 'scores_tmpdir')
    return results, scores

