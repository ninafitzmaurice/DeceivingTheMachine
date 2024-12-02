#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=09:00:00
#SBATCH --error=X_%j.err
#SBATCH --output=X_%j.out

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate cnnf

python train_exp.py --data 'data/UI128' \
                --exp-data-dir 'exp_data' \
                --train-cycles 1 \
                --exp-cycles 5 \
                --ind 5 \
                --mse-parameter 0.1 \
                --res-parameter 0.01 \
                --clean-parameter 0.05 \
                --lr 0.01 \
                --batch-size 14 \
                --val-batch-size 24 \
                --epochs 15 \
                --seed 666 \
                --grad-clip \
                --model-dir '1cyc_15epoch_NEWEST_res0.01'  \
                --model-name '1cyc_15epoch_exp_all_epoch'


echo "Finished <3"