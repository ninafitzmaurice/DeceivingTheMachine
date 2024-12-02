#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=18:00:00
#SBATCH --error=ADV_%j.err
#SBATCH --output=ADV_%j.out

. ~/.bashrc
module load 2021
module load Anaconda3/2021.05

conda activate cnnf
echo "Starting train_UI_adv.py"

python train_UI_adv.py --data 'data/UI128' \
                        --exp-data-dir 'exp_data' \
                        --train-cycles 1 \
                        --exp-cycles 5 \
                        --ind 5 \
                        --mse-parameter 0.1 \
                        --res-parameter 0.01 \
                        --clean-parameter 0.5 \
                        --lr 0.01 \
                        --batch-size 14 \
                        --val-batch-size 24 \
                        --epochs 10 \
                        --seed 666 \
                        --grad-clip \
                        --model-dir 'ADV_protocol_3'  \
                        --model-name '1cyc_10epoch'

echo "Finished"
