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
from detectron2.utils.visualizer import ColorMode


cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join("output/model_final.pth")  # path to the model we just trained

cfg.DATASETS.TEST = ("logos", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.03   # set a custom testing threshold
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

test_metadata = MetadataCatalog.get("logos")
from detectron2.utils.visualizer import ColorMode
import glob
for imageName in glob.glob('Flick/FlickrLogos-v2/classes/jpg/pepsi/*.jpg'):
    im = cv2.imread(imageName)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("outputs/%s.jpg" % str(imageName[-8:-5]), out.get_image()[:, :, ::-1])


# im = cv2.imread("images/input_N.jpg")
# im = cv2.imread("/home/tugdual/Documents/EPFL/Sticker/Flick/FlickrLogos-v2/classes/jpg/aldi/55601843.jpg")

# outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
# v = Visualizer(im[:, :, ::-1], scale=1)

# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imwrite("test.jpg", out.get_image()[:, :, ::-1])
# cv2.waitKey(0