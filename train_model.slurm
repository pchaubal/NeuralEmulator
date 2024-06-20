#!/bin/bash
#SBATCH --job-name="mltrain"
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=0-01:59:00

source /home/pchaubal/.bashrc

export OMP_NUM_THREADS=12
cd /home/pchaubal/ML/NeuralBoltzmann

conda activate tf

python src/dense.py
python src/retrain.py
python sampler/emcee_example.py
