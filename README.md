# NeuralBoltzmann

Code for training neural network models approximating Boltmann codes.

# Running

## Dependencies

See `requirements.txt`

```
pip install -r requirements.txt --user
```

## Training

Designed to run from base directory, e.g.:

```
python src/dense.py --mode train
```

## On NERSC

To get access to GPUs first apply for Perlmutter preparedness allocation (project m1759).

Then to get access to the GPU queue on Cori do:

```
module load cgpu
```

For `tensorflow-2.2.0`:

```
module load tensorflow/gpu-2.2.0-py37
```

And install any extra packages with `pip install <package> --user`.

## Hyperparameter Optimization

To run with `keras-tuner` we have to fix `keras-tuner==1.0.0`, as version 1.0.2 is incompatible with tensorflow 2.2. Can not upgrade to 2.4 yet on Cori easily.

Run all scripts from the root directory:

```
sbatch ./scripts/hyperparam_job.sh
```