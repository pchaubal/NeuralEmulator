#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=5
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=272
#SBATCH --constraint=knl
#SBATCH --output=./logs/camb-%j.out
#SBATCH --error=./logs/camb-%j.err

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export OMP_NUM_THREADS=68

export NEURALBOLTZMANN_DATA_DIR=/global/cscratch1/sd/bthorne/NeuralBoltzmann
module load python
srun python src/datasets.py --debug