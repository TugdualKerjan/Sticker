# Sticker detection using Faster-RCNN 🎉 

### Bachelors project at EPFL 2021

---

## Week 1 :

* Learning how to read scientific papers, took a look at [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/pdf/2003.06957v1.pdf) 
    * The abstract of scientific papers usually states the simple but obvious
    * The intro states what the problem is and what their approach to fixing it will be. In this problem they only modify the last layer, that is the Box Classifier and the Box Regressor.

* The previous cited paper uses Detectron 2, which implements Faster-RCNN in PyTorch - **This is likely what I will be using for this project** 

* [COCO](https://cocodataset.org/#home) is a dataset frequently used in the field of object detection 

* [Paperswithcode](https://paperswithcode.com/task/few-shot-object-detection) has an excellent database of possible reseach papers with implementations.

---

## Week 2

Although **RetinaNet** is proposal free and thus faster as it is just one CNN it lacks the modularity of the proposal based **Faster-RCNN** (Which is **Fast-RCNN** with a **RPN**).

Deep object cosegmentation takes two images and finds the common features, would be potentially efficient for comparing to existing stickers.



* [Few-shot Object Detection via Feature Reweighting](https://arxiv.org/pdf/1812.01866v2.pdf) Proposal based vs Proposal free: Uses loadable vectors to change the weights, with two Inputs, Meta-feature and LW reweighting

* [Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector](https://arxiv.org/pdf/1908.01998v4.pdf) No need to fine tune the model to novel classes - Uses way more catergories and few images per category, uses Attention network (Garbage in -> Garbage out). Concept of multirelation

* [Polytechnique X, MetaLearning algorithms](https://arxiv.org/pdf/1909.13579v1.pdf) Interesting paper with details of implementation

* [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf) The reference in Proposal based FSO detection
    * [Amazing blog post on the implementation of Mask R-CNN](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)
    * [Implementation of Mask R-CNN](https://github.com/matterport/Mask_RCNN) Sadly uses tensowflow

---

## Week 3

Installed Detectron2 in myenv environment, with PyTorch 1.7.1 and TorchVision 0.8.2, CPU version. Would like to connect to SCITAS and use that instead.

D2Go is interesting optimised version of Detectron2 but for mobile phones, gotta check it out.

* [Fantastic intro to detectron2](https://www.youtube.com/watch?v=EVtMT6Ve0sY)

* [Traffic sign detection](https://www.youtube.com/watch?v=SWaYRyi0TTs) Could be useful as similar to stickers

* [How to train detectron2 on a custom dataset](https://www.youtube.com/watch?v=CrEW8AfVlKQ)
    * [The blog](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch) Where it is explained in great detail


* Datasets:
    * [FlikrLogos](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/) Have to send email to get dataset ✔
    * [BelgaLogos](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html#download)

* A [mobile first version](https://github.com/facebookresearch/d2go) of Detectron2 which is light weight

### How to run detecton2 demo:

<details close>
<summary></summary>

- Install packages from [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

- Run after pulling the git

```terminal
git clone https://github.com/facebookresearch/detectron2.git
cd demo
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../../input.jpg --opts MODEL.DEVICE cpu MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
</details>


### Issues I ran into:
<details close>
<summary></summary>
- Had to add MODEL.DEVICE cpu for it to run on CPU

- Had to point to a downloaded image
```
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
```

- Had to install two libraries for OpenCV
```
pip install opencv
```
</details>

### What I learned

- How to do Markdown
- Why and how of Conda environments
- How to use detectron pretrained models

- Names of the pretrained

    - R50, R101 is [MSRA Residual Network](https://github.com/KaimingHe/deep-residual-networks)
    - X101 is ResNeXt
    - Use 3x as it is more trained than 1x

---

## Week 4

Managed to ssh into SCITAS, spent some time understanding how to access CUDA, got it to run!

Downloaded the FlickrLogo dataset [Links](notes/PRIVATE.md)!

Wrote a simple [script](execute.sh) that will execute as a job using Slurm, made a venv with these packages added user for it to not be system installed:

```terminal
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html --user
pip install torch==1.7.1 torchvision==0.8.2 --user
pip install opencv-python --user
pip install -U iopath==0.1.4 --user
```

To interact with SCITAS *and* have it on the GPUs

```terminal
Sinteract -t 00:10:00 -p gpu -q gpu_free -g gpu:1
```

### Issues I ran into:
<details close>
<summary></summary>

- Have to load python after the venv as venv replaces the python version

- Have to install opencv-python everytime even I'm in the venv?
```terminal
pip install opencv-python --user
```
- Had to downgrade iopath for it to work on SCITAS
```
pip install -U iopath==0.1.4 --user
```
</details>


### What I learned:

- How to use SCITAS again :P

- How to use scp and send the images back to my local machine

- How to launch jobs instead of using Sinteract

- What Python notebooks were and there use using the [detectron2 tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=8IRGo8d0qkgR)

---

## Week 5

Trained detectron2 on FlickrLogos32, which has 32 classes, each a company.

Detectron needs me to register a function that will return a list\[`dict`\] where the `dict` holds information about 1 specific image, then to register it into the datasets, which can then be passed to the __dataloader__ which will augment, batch and give to `model.forward()`

https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html

<details close>
<summary>Fields required for segmentation</summary>

Items like `file_name`, `height`, `width`, `image_id` can be given by

```python
record = {}

filename = os.path.join(img_dir, v["filename"])
height, width = cv2.imread(filename).shape[:2]

record["file_name"] = filename
record["image_id"] = idx
record["height"] = height
record["width"] = width
```

#### Since we are doing image segmentation

`annotations`, a list\[`dict`\] which contains `bbox`, `bbox_mode`, `category_id` and `segmentation` can be given by

```python
annos = v["regions"]
objs = []
for _, anno in annos.items():
    assert not anno["region_attributes"]
    anno = anno["shape_attributes"]
    px = anno["all_points_x"]
    py = anno["all_points_y"]
    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    poly = [p for x in poly for p in x]

    obj = {
        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": 0,
    }
    objs.append(obj)
record["annotations"] = objs
```

</details>

### Issues I ran into:
<details close>
<summary></summary>

- How to correctly open pictures (Had an annoying "\n" that was invisible in print() but not when passing as a path to open the image)

- How to correctly pass the mask (Have to transform it into __RLE__ which is lightweight binary mask)

</details>

### What I learned:

- The inner structure of detectron2, python (again)

- How to use `rsync`

- P a t i e n c e :stars:


# Works to cite:

Scalable Logo Recognition in Real-World Images
Stefan Romberg, Lluis Garcia Pueyo, Rainer Lienhart, Roelof van Zwol
ACM International Conference on Multimedia Retrieval 2011 (ICMR11), Trento, April 2011. 

