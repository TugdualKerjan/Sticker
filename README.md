# Sticker detection using Faster-RCNN 🎉 

### Bachelors project at EPFL 2021

---

## Week 1

* Learning how to read scientific papers, took a look at [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/pdf/2003.06957v1.pdf) 
    * The abstract of scientific papers usually states the simple but obvious
    * The intro states what the problem is and what their approach to fixing it will be. In this problem they only modify the last layer, that is the Box Classifier and the Box Regressor.

* The previous cited paper uses Detectron 2, which implements Faster-RCNN in PyTorch - **This is likely what I will be using for this project** 

* [COCO](https://cocodataset.org/#home) is a dataset frequently used in the field of object detection 

* [Paperswithcode](https://paperswithcode.com/task/few-shot-object-detection) has an excellent database of possible reseach papers with implementations.


## Week 2

Although **RetinaNet** is proposal free and thus faster as it is just one CNN it lacks the modularity of the proposal based **Faster-RCNN** (Which is **Fast-RCNN** with a **RPN**).

Deep object cosegmentation takes two images and finds the common features, would be potentially efficient for comparing to existing stickers.



* [Few-shot Object Detection via Feature Reweighting](https://arxiv.org/pdf/1812.01866v2.pdf) Proposal based vs Proposal free: Uses loadable vectors to change the weights, with two Inputs, Meta-feature and LW reweighting

* [Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector](https://arxiv.org/pdf/1908.01998v4.pdf) No need to fine tune the model to novel classes - Uses way more catergories and few images per category, uses Attention network (Garbage in -> Garbage out). Concept of multirelation

* [Polytechnique X, MetaLearning algorithms](https://arxiv.org/pdf/1909.13579v1.pdf) Interesting paper with details of implementation

* [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) The reference in Proposal based FSO detection
    * [Amazing blog post on the implementation of Mask R-CNN](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
    * [Implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN) Sadly uses tensowflow

## Week 3

* [Fantastic intro to detectron2](https://www.youtube.com/watch?v=EVtMT6Ve0sY)

* [Traffic sign detection](https://www.youtube.com/watch?v=SWaYRyi0TTs) Could be useful as similar to stickers

* [How to train detectron2 on a custom dataset](https://www.youtube.com/watch?v=CrEW8AfVlKQ)
    * [The blog](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch) Where it is explained in great detail


* Datasets:
    * [FlikrLogos](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/) Have to send email to get dataset ✔
    * [BelgaLogos](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html#download)

* A [mobile first version](https://github.com/facebookresearch/d2go) of Detectron2 which is light weight