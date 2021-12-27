import os.path as osp
import tempfile

import mmcv
import numpy as np
from PIL import Image

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module(force=True)
class CUBoxDataset(CustomDataset):
    """
    CUBox Dataset
    index 0번째가 배경(관심 없는 것들) 을 나타내도록 함
    총 200개 + 1(배경) 카테고리
    """

    CLASSES = (
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
               [59, 123, 183], [68, 112, 177], [76, 101, 172], [85, 90, 167]][1:201]

    def __init__(self, **kwargs):
        super(CUBoxDataset, self).__init__(
            img_suffix=".jpg", seg_map_suffix=".png", reduce_zero_label=True, **kwargs
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

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

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
