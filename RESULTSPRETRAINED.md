<details close>
<summary><h1>On personal computer</h1></summary>

## Input
![](images/input.jpg)

## Masking

<details close>
<summary>Terminal command</summary>

```
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../../input.jpg --output output.jpg --opts MODEL.DEVICE cpu MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
</details>

### Mask_rcnn_R_50_FPN_3x

15 instances in 3.19s

![](images/output_mask_rcnn_R_50_FPN_3x.jpg)

## Detection proposal-based

### Faster_rcnn_R_101_FPN_3x

17 instances in 4.23s

![](images/output_faster_rcnn_R_101_FPN_3x.jpg)

## Detection proposal-free

### Retinanet_R_101_FPN_3x

13 instances in 3.83s

![](images/output_retinanet_R_101_FPN_3x.jpg)

## Panoptic detection

### Panoptic_fpn_R_50_3x.yaml

![](images/output_panoptic_fpn_R_50_3x.jpg)

</details>

<details close>
<summary><h1>Using python</h1></summary>

## Input

![](images/input_N.jpg)

## Masking

### Mask_rcnn_R_101_FPN_3x

![](images/output_N_mask_rcnn_R_101_FPN_3x.jpg)

### Mask_rcnn_R_101_C4_3x

![](images/output_N_mask_rcnn_R_101_C4_3x.jpg)

### Mask_rcnn_R_101_DC5_3x

![](images/output_N_mask_rcnn_R_101_DC5_3x.jpg)

</details>

<details close>
<summary><h1>On SCITAS</h1></summary>


## Input
![](images/input.jpg)

## Masking


<details close>
<summary>Details</summary>
detected 15 instances in 0.65s

```
#SBATCH --time=0:10:0
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
```
</details>

### Mask_rcnn_R_50_FPN_3x

![](images/output_S_mask_rcnn_R_50_FPN_3x.jpg)

</details>
