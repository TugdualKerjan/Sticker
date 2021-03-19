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


path = "Flick/FlickrLogos-v2/"
classes = ["adidas","aldi","apple","becks","bmw","carlsberg","chimay","cocacola","corona","dhl","erdinger","esso","fedex","ferrari","ford","fosters","google","guiness","heineken","hp","milka","nvidia","paulaner","pepsi","rittersport","shell","singha","starbucks","stellaartois","texaco","tsingtao","ups"]

# Mention there is a bitmask and not polygon
cfg = get_cfg()
cfg.INPUT.MASK_FORMAT = 'bitmask'

def get_logos():
    dataset_dicts = []
    # for idx, v in enumerate(imgs_anns.values()):
    for line in open(path+"trainset.txt", "r"):
        imgclass, imgname = line.split(",")
        imgname = imgname[:-1] #remove extra \n
        imgclass = imgclass.lower() #Lower case HP
        record = {}
        
        filepath = os.path.join(path,"classes/jpg/",imgclass,imgname)
        height, width = cv2.imread(filepath).shape[:2]
        
        record["file_name"] = filepath
        record["image_id"] = imgname[:-4] #Remove the .jpg
        record["height"] = height
        record["width"] = width

        filepathmask = os.path.join(path,"classes/masks/",imgclass,imgname)

        bbox = open(filepathmask+".bboxes.txt", "r").readlines()[1].split(" ")

        b_a = np.array(cv2.imread(filepathmask+".mask.0.png")[:, :, 0] > 128, dtype=bool, order='F') # Already in grayscale, change to binary

        # Only one object per image for this one
        record["annotations"] = [{
                "bbox": [int(x) for x in bbox],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": pycocotools.mask.encode(b_a), #cfg.INPUT.MASK_FORMAT must be set to bitmask if using the default data loader with such format.
                "category_id": classes.index(imgclass),
        }]

        dataset_dicts.append(record)
    return dataset_dicts # Returns a dict of all images with their respective descriptions

# Registration
DatasetCatalog.register("logos", get_logos) # Registering the dataset
MetadataCatalog.get("logos").set(thing_classes=classes) # Adding metadata

# logo_metadata = MetadataCatalog.get("logos")
# dataset_dicts = get_logos()
# Take three random images, apply the masks and output them
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=logo_metadata)
#     out = visualizer.draw_dataset_dict(d)
#     result = out.get_image()[:, :, ::-1]
#     cv2.imwrite("output/output%s.jpg" % str(d["file_name"][-8:-6]), result)

from detectron2.engine import DefaultTrainer

# 
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("logos",) # Train with the logos dataset
cfg.DATASETS.TEST = () # No test
# cfg.MODEL.DEVICE = "cpu"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

print(cfg.OUTPUT_DIR)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()