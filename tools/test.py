import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from backbone import beit, beit_attn
import mmcv_custom
from mmcv_custom import encoder_decoder
from mmcv_custom import CUBOXDataset, CUBOXInstanceDataset
from segmentor_custom.encoder_decoder import EncoderDecoderAP
from mmcv_custom.test_with_logits import multi_gpu_test_logits, single_gpu_test_logits


"""
def logits_test(model, data_loader, distributed=False, device_ids=[], gpu_collect=False, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    print("Dataset length: ", len(dataset))
    for i, data in enumerate(data_loader):
        # print(data.keys()) # 결과: img_metas, img
        # print(data['img'][0].shape)
        # print(data['img_metas'])
        with torch.no_grad():
            # data 내 키우등는 imgs, img_metas가 포함되어 있음
            # kwargs = {k: v for k,v in data.items() if k not in ['imgs', 'img_metas']}
            kwargs = {
                'img': data['img'][0].cuda(),
                'img_meta': data['img_metas'][0].data[0]
            }
            # if 'imgs' in data:
                # kwargs['img'] = data['imgs'][0] # 무슨 aug 관련해서 형태가 꼬여 있는듯. forward_test, simple_test 함수 등 확인하면 이해 됨
            # if 'img_metas' in data:
                # kwargs['img_meta'] = data['img_metas'][0].data[0] # DataContainer라서 사용이 안됨.. 그래서 data를 불러왔음
            kwargs['rescale'] = True
            if device_ids:
                _, kwargs = model.to_kwargs([], kwargs, device_ids[0])
                logit_result = model.module.inference(**kwargs[0])
            else:
                logit_result = model.module.inference(**kwargs)
        
        if isinstance(logit_result, list):
            results.extend(logit_result)
        else:
            results.append(logit_result)
    
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)

    return results
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show_dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval_sample_iou', action='store_true')
    parser.add_argument('--inference_with_score', action='store_true', default=True)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.eval_sample_iou, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir" or "--eval_sample_iou"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    mask_scores =None
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        if args.inference_with_score:
            outputs_with_score = single_gpu_test_logits(model, data_loader, args.show, args.show_dir,
                                    efficient_test)
            outputs, mask_scores = outputs_with_score
        else:
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        device_ids= [torch.cuda.current_device()]
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=device_ids,
            broadcast_buffers=False)
        if args.inference_with_score:
            outputs_with_score = multi_gpu_test_logits(model, data_loader, args.tmpdir,
                                    args.gpu_collect, efficient_test)
            outputs, mask_scores = outputs_with_score
        else:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            if 'mAP' in args.eval:
                print("\nEvaluate with mAP!!")
                print("mask length", len(mask_scores)) # num_samples
                print("mask shape", mask_scores[0].shape) # 201
                print("pred shape", outputs[0].shape) # 201
            dataset.evaluate(outputs, metric=args.eval, mask_scores=mask_scores, **kwargs)
        if args.eval_sample_iou:
            dataset.iou_single(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    main()
