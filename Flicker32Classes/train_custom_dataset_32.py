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


path = "Flick/FlickrLogos-v2/"
classes = ["adidas","aldi","apple","becks","bmw","carlsberg","chimay","cocacola","corona","dhl","erdinger","esso","fedex","ferrari","ford","fosters","google","guiness","heineken","hp","milka","nvidia","paulaner","pepsi","rittersport","shell","singha","starbucks","stellaartois","texaco","tsingtao","ups"]

# Mention there is a bitmask and not polygon

def get_logos(directory):
    dataset_dicts = []

    for line in open(directory, "r"):
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

        if(imgclass == "no-logo"):
            record["annotations"] = []
        else:
            filepathmask = os.path.join(path,"classes/masks/",imgclass,imgname)

            bbox = open(filepathmask+".bboxes.txt", "r").readlines()[1].split(" ")

            b_a = np.asarray(cv2.imread(filepathmask+".mask.0.png")[:, :, 0] == 255, dtype=bool, order='F') # Already in grayscale, change to binary
            # Only one object per image for this one
            record["annotations"] = [{
                    "bbox": [int(x) for x in bbox],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": pycocotools.mask.encode(b_a), #cfg.INPUT.MASK_FORMAT must be set to bitmask if using the default data loader with such format.
                    "category_id": classes.index(imgclass),
                    # "category_id": 0,
            }]

        dataset_dicts.append(record)
    return dataset_dicts # Returns a dict of all images with their respective descriptions


for d in ["train", "test"]:
    DatasetCatalog.register("logo_" + d, lambda d=d: get_logos(path + d + "set.txt"))
    MetadataCatalog.get("logo_" + d).set(thing_classes=classes)

logo_metadata = MetadataCatalog.get("logo_train")
dataset_dicts = DatasetCatalog.get("logo_train")

# Take three random images, apply the masks and output them
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=logo_metadata)
#     out = visualizer.draw_dataset_dict(d)
#     result = out.get_image()[:, :, ::-1]
#     cv2.imwrite("output/output%s.jpg" % str(d["file_name"][-8:-6]), result)

model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

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
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()