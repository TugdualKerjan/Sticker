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


path = "../Flick/FlickrLogos-v2/"

def get_logos(directory):
    dataset_dicts = []

    for line in open(directory, "r"):
        imgclass, imgname = line.split(",")
        imgname = imgname[:-1] #remove extra \n
        imgclass = imgclass.lower() #Lower case HP
        record = {}
        print(imgname)
        
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
                    "category_id": 0,
            }]

        dataset_dicts.append(record)
    return dataset_dicts # Returns a dict of all images with their respective descriptions

# DatasetCatalog.register("logo_test", lambda: get_logos(path + "testset.txt"))
# dataset_dicts = DatasetCatalog.get("logo_test")

MetadataCatalog.get("logo_test").set(thing_classes=["logo"])
logo_metadata = MetadataCatalog.get("logo_test")

model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.INPUT.MASK_FORMAT = 'bitmask'
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (logo). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.DATASETS.TEST = ("logo_test", )
predictor = DefaultPredictor(cfg)

os.makedirs("guess", exist_ok=True)

import glob
for imageName in random.sample(glob.glob('../Flick/FlickrLogos-v2/classes/jpg/*/*.jpg'), 30):
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=logo_metadata, 
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("guess/%s.jpg" % str(imageName[-8:-5]), out.get_image()[:, :, ::-1])

# dataset_dicts = get_logos('../Flick/FlickrLogos-v2/testset.txt')

# for d in random.sample(dataset_dicts, 10):    
#     print(d["file_name"])
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=logo_metadata, 
#                    scale=0.8, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imwrite("guess/%s.jpg" % str(d["file_name"][-8:-5]), out.get_image()[:, :, ::-1])

# cfg.DATASETS.TRAIN = ("logo_train",) # Train with the logos dataset
# cfg.DATASETS.TEST = () # No test
# cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 8
# cfg.SOLVER.BASE_LR = 0.02  # pick a good LR
# cfg.SOLVER.MAX_ITER = 600    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)