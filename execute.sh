#!/bin/bash -l
#SBATCH --time=0:10:0
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Sinteract -t 00:10:00 -p gpu -q gpu_free -g gpu:1
 
# ? Not sure what this does slmodules -s x86_E5v2_Mellanox_GPU
# module load gcc cuda cudnn mvapich2 openblas
# source venv-tensorflow-1.9/bin/activate
# srun python your_input.py


module load gcc/8.4.0-cuda cuda/10.2.89 && source sticker/env/bin/activate && module load python/3.7.7

# Load the python
module load gcc/8.4.0-cuda cuda/10.2.89
# Open the venv
source sticker/env/bin/activate
# Load python
module load python/3.7.7

# Issue with iopath
pip install -U iopath==0.1.4 --user

# Go into the dir
cd detectron2/demo/

# Execute
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.jpg --output output.jpg --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl