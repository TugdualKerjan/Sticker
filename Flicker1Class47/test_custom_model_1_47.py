import random
import cv2
import json
import os
import glob
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import torch
import detectron2
import pycocotools
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries

# import some common detectron2 utilities


path = "../Flick47/"

def get_logos(directory):
    dataset_dicts = []

    for line in open(directory + "/filelist.txt", "r"):
        imgname = line[2:-1] #remove extra \n and ./
        record = {}
        
        filepath = os.path.join(directory,imgname)
        # print(filepath)
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
                # cv2.imshow("a", imgmask)
                # cv2.waitKey(0)
                b_a = np.asarray(imgmask[:, :, 0] == 255, dtype=bool, order='F') # Already in grayscale, change to binary

                record["annotations"].append({
                        "bbox": [int(x) for x in [x1, y1, x2, y2]],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": pycocotools.mask.encode(b_a), #cfg.INPUT.MASK_FORMAT must be set to bitmask if using the default data loader with such format.
                        "category_id": 0,
                })
        
        dataset_dicts.append(record)
    return dataset_dicts # Returns a dict of all images with their respective descriptions


for d in ["test"]:
    DatasetCatalog.register("logo_" + d, lambda d=d: get_logos(path + d))
    MetadataCatalog.get("logo_" + d).set(thing_classes=["logo"])

MetadataCatalog.get("logo_test").set(thing_classes=["logo"])
logo_metadata = MetadataCatalog.get("logo_test")

model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.MODEL.DEVICE = "cpu"
# only has one class (logo). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.DATASETS.TEST = ("logo_test", )
predictor = DefaultPredictor(cfg)

os.makedirs("guess", exist_ok=True)

for imageName in random.sample(DatasetCatalog.get("logo_test"), 30):
# for imageName in glob.glob('../poly.jpg'):
    im = cv2.imread(imageName["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=logo_metadata,
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("guess/%s.jpg" % str(imageName["file_name"][-10:-3]), out.get_image()[:, :, ::-1])