import torch
import detectron2
import pycocotools
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

path = "Flick47/"

def get_logos(directory):
    dataset_dicts = []

    for line in open(directory + "/filelist-logosonly.txt", "r"):
        imgname = line[2:-1] #remove extra \n and ./
        record = {}
        
        filepath = os.path.join(directory,imgname)
        height, width = cv2.imread(filepath).shape[:2]
        
        record["file_name"] = filepath
        record["image_id"] = imgname[7:-4] #Remove the .png and 000001/
        record["height"] = height
        record["width"] = width

        record["annotations"] = []

        if "no-logo" not in imgname:
            filepathmask = os.path.join(directory,imgname[:-4]+".gt_data.txt")
            for anno in open(filepathmask):
                x1, y1, x2, y2, imgclass, _, mask, _, _ = anno.split(" ")


                imgmask = cv2.imread(directory+"/"+imgname[:-4]+"."+str(mask)+".png")
                b_a = np.asarray(imgmask[:, :, 0] == 255, dtype=bool, order='F') # Already in grayscale, change to binary

                record["annotations"].append({
                        "bbox": [int(x) for x in [x1, y1, x2, y2]],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": pycocotools.mask.encode(b_a), #cfg.INPUT.MASK_FORMAT must be set to bitmask if using the default data loader with such format.
                        "category_id": 0,
                })
        
        dataset_dicts.append(record)
    return dataset_dicts # Returns a dict of all images with their respective descriptions


for d in ["train"]:
    DatasetCatalog.register("logo_" + d, lambda d=d: get_logos(path + d))
    MetadataCatalog.get("logo_" + d).set(thing_classes=["logo"])

test_metadata = MetadataCatalog.get("logo_train")

model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.DATASETS.TRAIN = ("logo_train",) # Train with the logos dataset
cfg.DATASETS.TEST = () # Train with the logos dataset

cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
cfg.SOLVER.MAX_ITER = 2000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (logo). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()