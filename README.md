# Sticker detection using Faster-RCNN 🎉 

## Bachelors project at EPFL 2021

---

## Week 1

* Learning how to read scientific papers, took a look at [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/pdf/2003.06957v1.pdf) 
    * The abstract of scientific papers usually states the simple but obvious
    * The intro states what the problem is and what their approach to fixing it will be. In this problem they only modify the last layer, that is the Box Classifier and the Box Regressor.

* The previous cited paper uses Detectron 2, which implements Faster-RCNN in PyTorch - **This is likely what I will be using for this project** 

* [COCO](https://cocodataset.org/#home) is a dataset frequently used in the field of object detection 

* [Paperswithcode](https://paperswithcode.com/task/few-shot-object-detection) has an excellent database of possible reseach papers with implementations.