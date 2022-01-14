import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

from mmseg.core.evaluation.metrics import intersect_and_union
from functools import reduce
from mmseg.core import eval_metrics

# cubox_dataset에 붙여 넣으면 좋을 코드
import tempfile
import os.path as osp
from collections import OrderedDict, defaultdict
import io
import contextlib
import itertools
import copy

from terminaltables import AsciiTable

from mmcv.utils import print_log
# from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmcv_custom.mmdet.api_wrappers import COCO, COCOeval
import pycocotools.mask as mask_util


@DATASETS.register_module(force=True)
class CUBOXInstanceDataset(CustomDataset):
    """
    CUBox Dataset
    mAP로 평가가 가능하도록 일부 변경한 것
    index 0번째가 배경(관심 없는 것들) 을 나타내도록 함
    총 200개 + 1(배경) 카테고리
    """

    CLASSES = (
        "background",
        "AirFryer",
        "AirtightContainer",
        "Apple",
        "Backpack",
        "Banana",
        "BandAid",
        "BathMat",
        "BeanCurd",
        "BeerBottle",
        "BellPepper",
        "Bicycle",
        "Blender",
        "Book",
        "Broccoli",
        "Broom",
        "Brush",
        "Bucket",
        "Cabbage",
        "Calculator",
        "Camera",
        "Can",
        "CarSideMirror",
        "CarWheel",
        "Cardigan",
        "Carrot",
        "Cereal",
        "Chair",
        "Cheese",
        "ChoppingBoard",
        "Cigarettes",
        "Clock",
        "Coat",
        "ComputerKeyboard",
        "ComputerMouse",
        "CookingOil",
        "Corn",
        "Cucumber",
        "Cucurbita",
        "DeskLamp",
        "DesktopComputer",
        "DigitalDoorlock",
        "DiningTable",
        "DishDrainer",
        "DishSoap",
        "DisposableMask",
        "Doll",
        "DoorHandle",
        "Dumbbell",
        "ElectricFan",
        "ElectricKettle",
        "ElectricRiceCooker",
        "ElectricSwitch",
        "Envelope",
        "EspressoMaker",
        "Eyepatch",
        "FacePowder",
        "Fan",
        "Flashlight",
        "FlowerPot",
        "FluorescentLamp",
        "Fork",
        "Frame",
        "Frypan",
        "FuseBox",
        "HairBand",
        "HairBrushComb",
        "HairDryer",
        "HairSpray",
        "Hamburger",
        "Hammer",
        "Handbag",
        "Hanger",
        "Hankerchief",
        "Hat",
        "Headphone",
        "Hotdog",
        "Hourglass",
        "HwatuCard",
        "Icecream",
        "InstantRice",
        "Iron",
        "Jug",
        "Ketchup",
        "Kleenex",
        "Knife",
        "Ladle",
        "LaptopComputer",
        "LaundryDetergent",
        "Lemon",
        "Loaf",
        "Loafer",
        "Lotion",
        "Macaron",
        "Magnifier",
        "Mailbag",
        "Mailbox",
        "ManholeCover",
        "Microphone",
        "Microwave",
        "Milk",
        "Mitten",
        "MixingBowl",
        "MobilePhone",
        "Monitor",
        "Mop",
        "Mosquitocide",
        "MotorScooter",
        "Muffler",
        "Mug",
        "Mushroom",
        "One-piece",
        "Orange",
        "Pants",
        "PaperBag",
        "PaperNotebook",
        "ParkBench",
        "Pen",
        "PencilCase",
        "PencilSharpener",
        "PennyBank",
        "Perfume",
        "PillBottle",
        "Pillow",
        "Pizza",
        "Plate",
        "Plunger",
        "Pole",
        "PopBottle",
        "Popsicle",
        "Pot",
        "Potato",
        "Printer",
        "Projector",
        "Puncher",
        "Ramen",
        "Razor",
        "RemoteControl",
        "RiceScoop",
        "RiceSoupBowl",
        "RubberEraser",
        "Ruler",
        "RunningShoe",
        "SaltShaker",
        "Sandal",
        "Scale",
        "Scissors",
        "ScouringPad",
        "Screwdriver",
        "SesameOil",
        "SewerDrainage",
        "Shampoo",
        "ShoppingCart",
        "ShowerHead",
        "Skirt",
        "Slippers",
        "Soap",
        "Sock",
        "Spatula",
        "Speaker",
        "Spoon",
        "SportsBall",
        "Sprayer",
        "Stapler",
        "Stove",
        "Suitcase",
        "Sunglasses",
        "Sunscreen",
        "SupplementaryBattery",
        "Sweatshirt",
        "T-shirt",
        "TabletPC",
        "Tape",
        "Teacup",
        "Teapot",
        "Television",
        "Thermometer",
        "Tie",
        "Toaster",
        "Toilet",
        "ToiletTissue",
        "Tomato",
        "Toolkit",
        "Toothbrush",
        "Toothpaste",
        "Towel",
        "Tray",
        "USB",
        "Umbrella",
        "Vase",
        "Wallet",
        "Washbasin",
        "Wastebin",
        "Watch",
        "WaterBottle",
        "WaterPurifier",
        "WetTissue",
        "Whisk",
        "WineBottle",
        "WineGlass",
        "Wok",
    )
 
    PALETTE = [[0, 0, 0], 
               [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255], [168, 12, 68], [179, 24, 71], 
               [190, 36, 73], [201, 48, 76], [211, 60, 78], [218, 70, 76], 
               [224, 79, 74], [230, 88, 72], [236, 97, 69], [242, 107, 67], 
               [245, 119, 71], [247, 131, 77], [248, 144, 83], [250, 157, 89], 
               [252, 170, 95], [253, 180, 102], [253, 190, 110], [253, 200, 119], 
               [253, 210, 127], [253, 220, 135], [254, 227, 145], [254, 233, 155], 
               [254, 239, 165], [254, 245, 175], [254, 251, 185], [252, 254, 187], 
               [247, 252, 179], [242, 250, 171], [237, 248, 164], [232, 246, 156], 
               [225, 243, 152], [213, 238, 155], [202, 233, 157], [190, 229, 160], 
               [179, 224, 162], [166, 219, 164], [153, 214, 164], [139, 208, 164], 
               [126, 203, 164], [112, 198, 164], [99, 191, 165], [89, 180, 170], 
               [79, 168, 175], [69, 157, 180], [59, 146, 184], [50, 134, 188], 
               [59, 123, 183], [68, 112, 177], [76, 101, 172], [85, 90, 167]][0:201]

    def __init__(self, **kwargs):
        super(CUBOXInstanceDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", reduce_zero_label=False, **kwargs
        )

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.
        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission
        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]

            filename = self.img_infos[idx]["filename"]
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f"{basename}.png")

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for ade20k evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), (
            "The length of results is not equal to the dataset len: "
            f"{len(results)} != {len(self)}"
        )

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None

        result_files = self.results2img(results, imgfile_prefix, to_label_id)
        return result_files, tmp_dir

    def iou_single(self,
                 results,
                 metric='IoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.

        """
        # print(self.ann_dir)
        # for img_info in self.img_infos:
        #     print(img_info['ann']['seg_map'])

        gt_seg_maps = self.get_gt_seg_maps(efficient_test)

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        
        num_imgs = len(results)
        assert len(gt_seg_maps) == num_imgs
        total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
        total_area_union = np.zeros((num_classes, ), dtype=np.float)
        total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
        total_area_label = np.zeros((num_classes, ), dtype=np.float)
        
        print("\n\n=== Per Sample Ground Truth IoU Score ===")
        for i in range(num_imgs):
            area_intersect, area_union, area_pred_label, area_label = \
                intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                    self.ignore_index, self.label_map, self.reduce_zero_label)
            single_gt_label = np.unique(gt_seg_maps[i])[np.nonzero(np.unique(gt_seg_maps[i]))]
            # single_gt_label = np.nonzero(np.unique(orig_gt_seg_maps[i]))
            seg_name = self.img_infos[i]['ann']['seg_map']
            print(f"Sample iou for {seg_name}: ", area_intersect[single_gt_label]/ area_union[single_gt_label]) #  한 개의 클래스에 대한 iou 출력
    
    def get_mask_score(self, segm_result, seg_logit):
        """
        - encoder_decoder 모델의 seg_logit 추론 결과를 받아서, 각 mask 별 = 한 클래스 당, 이미지 안에서 mask 한 개라고 취급
            mask score를 구한다
        - 차원(추측): (N, K(class), H, W) 적어도 K 부분이 class 라는 것은 확실한 듯 하다 
        - seg_results: segmentation label로 표현을 한 것. seg_pred = seg_logit.argmax(dim=1)
        - 주의사항: background는 계산에 포함하지 말 것
        수정!!
        - segm_result: True, False binary mask
        - seg_logit: logit values
        역할: segm_result 추론 영역에 해당하는 logit 값들의 평균을 구함
        """
        return seg_logit[segm_result].mean()

    def segmaps_to_encoded_instance(self, total_mask_scores, total_seg_preds):
        """
        전체 데이터를 받아서 포매팅 해주는 것
        """
        # print('log', len(total_seg_logits))
        # print('seg', len(total_seg_preds))
        # print('sel', len(self))
        assert len(total_mask_scores) == len(total_seg_preds) == len(self)
        results = [
                    self.segmap_to_encoded_instance(
                            total_mask_scores[idx],
                            total_seg_preds[idx])
                    for idx in range(len(total_mask_scores))
                ]
        return results

    # def segmap_to_encoded_instance(self, seg_logits, seg_preds):
    def segmap_to_encoded_instance(self, mask_scores, seg_preds):
        """
        이거 하나의 샘플에 대해서만 해야 함!!!!!! 주의주의
        하나의 이미지 당 많은 seg_pred가 존재할 수 있기 때문에 이렇게 되는 것이다
        encode_mask_results + single/multi-gpu-test
        mask_scores: (n_classes,) 차원, 각 클래스에 대한 모델의 예측값의 confidence
        """
        print("Self length", len(self))
        print("Seg length", len(seg_preds))
        print("Mask confidence", len(mask_scores))
        # if len(seg_preds.shape) == 2: # (w, h) 차원으로 단 한 개의 prediction만 있는 경우 = 항상!
            # print("Single result per batch!")
            # seg_preds = [seg_preds]
        # if len(mask_scores.shape) == 1:
            # mask_scores = [mask_scores]
        seg_preds = [seg_preds]
        mask_scores = [mask_scores]

        # test 추론 갯수 만큼 진행
        cls_segms = [[] for _ in range(len(self.CLASSES) - 1)]
        encoded_mask_results = [[] for _ in range(len(self.CLASSES) - 1)]
        cls_mask_scores = [[] for _ in range(len(self.CLASSES) - 1)]
        for idx in range(len(seg_preds)): # 한 샘플 내에 여러 개의 prediction
            seg = seg_preds[idx]
            labels = np.unique(seg) # 추측에서 나온 라벨들
            for label in labels:
                if label ==  0: continue # background는 계산하지 않음
                # cls_segm = seg[seg == label]
                # print(cls_segm.shape)
                cls_segm = seg == label
                # print(cls_segm.shape)
                # mask_score = self.get_mask_score(cls_segm, seg_logits[idx])
                mask_score = mask_scores[idx][label]
                # cls_segms[label - 1].append(cls_segm)# background 무시
                cls_segms[label - 1].append(cls_segm)
                cls_mask_scores[label - 1].append(mask_score)
                encoded_mask_results[label - 1].append(mask_util.encode(
                    np.array(cls_segm[:, :, np.newaxis], order='F', dtype='uint8'))[0])
        return encoded_mask_results, cls_mask_scores

    # def format_json(self, encoded_mask_results, cls_mask_scores):
    def format_seg_to_json(self, results):
        # assert len(encoded_mask_results) == len(cls_mask_scores) == len(self.CLASSES) - 1 # ignore background
        assert len(results) == len(self) # TODO 이거 mmseg에서도 len(self)를 했을 떄 이렇게 나오는 게 맞는지 확인
        segm_json_results = []
        for idx in range(len(results)):
            img_id = self.img_ids[idx] # TODO mmseg dataset도 이렇게 저장하고 있는지 확인해야 함
            seg = results[idx]

            for label in range(len(self.CLASSES) - 1):
                assert isinstance(seg, tuple)
                segms = seg[0][label]
                mask_score = seg[1][label]

                for i in range(len(segms)):
                    data = dict()
                    data['image_id'] = img_id
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label] # TODO cat_ids
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        
        return segm_json_results
    
    def format_instanceseg_result(self, results):
        result_files = dict()
        tmp_dir = tempfile.TemporaryDirectory()
        jsonfile_prefix = osp.join(tmp_dir.name, 'results')

        segm_json_results = self.format_seg_to_json(results)
        result_files['segm'] = f'{jsonfile_prefix}.segm.json'
        mmcv.dump(segm_json_results, result_files['segm'])

        return result_files, tmp_dir

    def eval_map(self, 
                mask_scores, 
                seg_preds, 
                metric='segm',
                logger=None,
                iou_thrs=None, 
                iou_type='segm', 
                proposal_nums=(100, 300, 1000),
                classwise=True
    ) -> None:
        """
        - mask_scores: list of predicted 
        """
        if not getattr(self, 'coco_gt', None):
            self.creat_coco_from_scratch()
        coco_gt = self.coco_gt
        #TODO img_ids refactor, 정확히 어떤 정보를 담고 있는 것인지 확인 필요
        # cat_ids = coco_gt.getCatIds(cat_names=self.CLASSES[1:]) # background 클래스를 무시한다
        # img_ids = coco_gt.get_img_ids()
        # cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        # metrics = ['segm'] # unused
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

        eval_results = OrderedDict()

        # TODO check again! === copy & paste region up & down
        results = self.segmaps_to_encoded_instance(mask_scores, seg_preds)
        result_files, tmp_dir = self.format_instanceseg_result(results)
        
        # TODO refactor 어차피 mertic segm만 볼거임
        predictions = mmcv.load(result_files['segm'])
        coco_dt = coco_gt.loadRes(predictions)

        cocoEval = COCOeval(coco_gt, coco_dt, iou_type)
        cocoEval.params.catIds = self.cat_ids
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }


        # ann_file 안에 img_ids에 대한 정보도 있어야 하는 거 같음
        cocoEval.evaluate()
        cocoEval.accumulate()

        # Save coco summarize print information to logger
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        print_log('\n' + redirect_string.getvalue(), logger=logger)

        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(self.cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                nm = coco_gt.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{nm["name"]}', f'{float(ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger=logger)

        metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
        ]

        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = cocoEval.stats[:6]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            f'{ap[4]:.3f} {ap[5]:.3f}')
        
        if tmp_dir is not None:
            tmp_dir.cleanup()
        
        return eval_results

    def creat_coco_from_scratch(self):
        coco = COCO()
        dataset = {}
        dataset['categories'] = [{'name': cate, 'id': i} for i, cate in enumerate(self.CLASSES[1:])]
        dataset['images'] = copy.deepcopy(self.img_infos)
        print(dataset['images'])
        gts = []
        for i, img_info in enumerate(dataset['images']):
            img = Image.open(osp.join(self.ann_dir, img_info['ann']['seg_map']))
            w, h = img.size
            img_info['width'] = w
            img_info['height'] = h
            img_info['id'] = i
            dataset['images'][i] = img_info
            seg_gt = np.array(img)
            gt = self.segmap_to_encoded_instance([1 for _ in range(len(self.CLASSES))], seg_gt)
            gts.append(gt)
        coco.imgs = {img['id']: img for img in dataset['images']}
        coco.cats = {cat['id']: cat['name'] for cat in dataset['categories']}
        coco.dataset = copy.deepcopy(dataset) # TODO
        
        self.cat_ids = coco.get_cat_ids(cat_names=self.CLASSES[1:]) # background 클래스를 무시한다
        self.img_ids = coco.get_img_ids()

        # TODO check again! === copy & paste region up & down
        gt_dump_files, tmp_dir = self.format_instanceseg_result(gts)
        gt_loaded = mmcv.load(gt_dump_files['segm'])

        self.coco_gt = coco.loadRes(gt_loaded)


    def evaluate(self,
                 results,
                 mask_scores=None,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            mask_scores(list): Mask confidency scores (calculated from logits) of the dataset. per class results are saved

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mAP']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            [m for m in metric if m != 'mAP'],
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric if m != 'mAP'] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)

        if 'mAP' in metric:
            print("Calculating mAP")
            map = self.eval_map(mask_scores, results, metric='segm',logger=None, iou_thrs=None, iou_type='segm', proposal_nums=(100, 300, 1000),classwise=True)
            print("mAP: ", map)
        return eval_results