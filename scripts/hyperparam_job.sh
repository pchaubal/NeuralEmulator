#!/bin/bash
#SBATCH -A m1759
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 10
#SBATCH --gpus-per-task=1
#SBATCH --output=./logs/hyper_search-%j.out
#SBATCH --error=./logs/hyper_search-%j.err

export SLURM_CPU_BIND="cores"
export NEURALBOLTZMANN_DATA_DIR=/global/cscratch1/sd/bthorne/NeuralBoltzmann

module load tensorflow/gpu-2.2.0-py37
module load texlive
srun python src/dense.py --mode hyper_search