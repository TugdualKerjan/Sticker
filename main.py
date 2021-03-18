import torch
import detectron2
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

im = cv2.imread("images/input_N.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

masks = outputs["instances"].pred_masks.cpu().numpy()
boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

w, h, _ = im.shape
TenthOfImage = w*h/10

for x in range(0,len(outputs["instances"].pred_masks)):
  boxtemp = boxes[x].astype(int)
#   if (boxtemp[3]-boxtemp[1])*(boxtemp[2]-boxtemp[0]) <TenthOfImage: continue
  masktemp = masks[x]

  result = im * masktemp[..., None]
  result = result[boxtemp[1]:boxtemp[3],boxtemp[0]:boxtemp[2]]
  cv2.imwrite("output/output%s.jpg" % str(x), result)
  
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imwrite("output.jpg", out.get_image()[:, :, ::-1])