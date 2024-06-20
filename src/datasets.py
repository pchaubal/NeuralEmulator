import os
from pathlib import Path

from absl import app
from absl import flags
from absl import logging

import gin

import numpy as np

from mpi4py import MPI 

import camb

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

DATA_DIR = Path(os.environ["NeuralBoltzmann_DATA"])
MODEL_DIR = Path(os.environ["NeuralBoltzmann_MODEL"])


class ScaledSpace():
    def __init__(self, param_scaling=None,scaling='linear',
                 spectra_path='./src/planck_2018_lensedCls.dat', scale=500, lpivot=900,lmax=2508):
        self.lpivot=int(lpivot)
        self.lmax=int(lmax)
        self.scale=np.abs(float(scale))
        self.scaling=scaling
        if self.scaling is None:
            self.lscaling = None
        elif self.scaling == 'linear':
            self.lscaling = 0.5/self.scale
        elif self.scaling == 'nonlinear':
            self.lscaling = np.ones(self.lmax-1)
            self.lscaling[self.lpivot-2:] = ( np.arange(self.lpivot,self.lmax+1) / self.lpivot )**2
            self.lscaling /= (2*self.scale)
        else:
            raise ValueError("NormalizeSpectra needs scaling to be 'linear' or 'nonlinear' or None.")
        
        self.param_scaling=param_scaling
        if self.param_scaling is None:
            #As, ns, ty,ombh2,omch2,theta
            parameter_names = np.asarray(['As','ns','tau','ombh2','omch2','theta'])
            column_max = np.asarray([3.0e-9, 1.1, 0.15, 0.025, 0.15, 1.045 ])
            column_min = np.asarray([1.5e-9, .90, 0.03, 0.019, 0.10, 1.035 ])
        else:
            column_min, column_max, parameter_names = param_scaling
        assert all(column_max > column_min)
        assert column_max.shape == column_min.shape
        self.column_min=column_min
        self.column_max=column_max
        self.parameter_names=parameter_names
        
        self.fiducial_spectrum = np.loadtxt(spectra_path)[:lmax-1,1]
        return

    def NormalizeParams(self, raw_params):
        """ Returns cosmological parameters normalized between 0 and 1"""
        normalized_params = (raw_params -self.column_min)/ (self.column_max - self.column_min)
        return normalized_params

    def DenormalizeParams(self, normalized_params,parameter_labels=None):
        if parameter_labels is not None:
            assert all(parameter_labels == self.parameter_names)
            # have not yet implemented reordering of parameter indices. future feature
        return (normalized_params + self.column_min) * (self.column_max - self.column_min)

    def NormalizeSpectra(self, spectra):
        if self.lscaling is None:
            return spectra
        assert spectra.shape[1] == self.lmax-1 # not dealing with different sizes yet. future feature
        return (spectra - (self.fiducial_spectrum) ) * self.lscaling + 0.5
            
    def DenormalizeSpectra(self, normalized_spectra):
        if self.lscaling is None:
            return normalized_spectra
        assert normalized_spectra.shape[1] == self.lmax-1 # not dealing with different sizes yet. future feature
        return (normalized_spectra - 0.5)/self.lscaling + self.fiducial_spectrum 


    def MakePrediction( self, params, model):
        """ Accepts raw params and gives back predicted spectra """
        nparams = self.NormalizeParams(params)
        predictions = model.predict(nparams)
        return self.DenormalizeSpectra(predictions)

def LogAndNormalizeSpectra(spectra):
    log_spectra = np.log10(spectra)
    # don't scale per column for spectra to maintain shape
    log_spectra_max = np.max(log_spectra)
    log_spectra_min = np.min(log_spectra)
    log_scaled_0_1_spectra = (log_spectra - log_spectra_min) / (log_spectra_max - log_spectra_min)
    return log_scaled_0_1_spectra

@gin.configurable
def LoadDataset(label='prakrut', ntrain=7000, output_dim=2550, scaling='linear' ):
    """ Return dataset as dictionary, containing train, test, validation splits, as well as 
    some metadata about scalings used so that we can convert to and from dimensionlful units
    later.

    ntrain will be specified, and then the remaining samples will be split between test and
    validation data.
    """
    if label == 'prakrut':
        spectra = np.load(DATA_DIR / "lcdmcl_tt_small.npy")[:, 2:2 + output_dim] # remove first two multipoles
        params = np.load(DATA_DIR  / "params.npy")[:50_000]
    if label == 'cobaya':
        spectra = np.load( DATA_DIR / "lcdm_dls.npy")[:,2:]
        params = np.load( DATA_DIR / "params.npy")[:,:6]
        parameter_names = np.asarray(['log_As','ns','theta','ombh2','omch2','tau'])
        column_max = np.asarray([3.5, 1.1, 1.045, 0.025, 0.15, 0.15 ])
        column_min = np.asarray([2.5, .90, 1.035, 0.019, 0.10, 0.03 ])
        param_scaling = [ column_min, column_max, parameter_names ]
    
    # Scale params by initializing the scaling class
    scalingspace = ScaledSpace(param_scaling = param_scaling, scaling=scaling)
    scaled_params = scalingspace.NormalizeParams(params)
    scaled_spectra = scalingspace.NormalizeSpectra(spectra)
    nsamples = spectra.shape[0]
    ntest = (nsamples - ntrain) // 2
    nval = nsamples - ntrain - ntest
    
    assert ntrain + ntest + nval == nsamples

    # start assembling return dataset
    dset = {}
    dset["nsamples"] = nsamples
    dset["ntrain"] = ntrain
    dset["ntest"] = ntest
    dset["nval"] = nval

    # record scaling relationships used in normalization
    dset["scalingspace"]=scalingspace


    # split up rescaled inputs / outputs
    split_inds = [ntrain, ntrain + ntest]
    dset["train"], dset["test"], dset["val"] = zip(np.split(scaled_params, split_inds), np.split(scaled_spectra, split_inds))

    # split up rescaled inputs / outputs
    dset["raw_train"], dset["raw_test"], dset["raw_val"] = zip(np.split(params, split_inds), np.split(spectra, split_inds))

    # check the logic was correct
    print(  )
    assert dset["train"][0].shape[0] == ntrain
    assert dset["test"][0].shape[0] == ntest
    assert dset["val"][0].shape[0] == nval

    return dset


def CAMBPowerspectrum(params, lmax=2550):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params[5], ombh2=params[3], omch2=params[4], mnu=0.06, omk=0, tau=params[2])
    pars.InitPower.set_params(As=params[0], ns=params[1], r=0)
    # set accuracy to recover up to lmax, and to get lensing potential right (do we need that?)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit='muK')
    return powers['total']

@gin.configurable
def CalculatePowerspectra(data_dir, As=[2.0e-9, 2.2e-9], ns=[.90,.99], 
        taus=[0.04,.08], omega_bs=[0.021, 0.023], omega_cs=[0.11,0.13], 
        H0s=[50,95], n_total=32, lmax=2500):    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        assert (n_total % size) == 0
    except AssertionError:
        raise AssertionError("Number of power spectra"
                "must be divisble by number of processes")

    # number of spectra to calculate on each process    
    n_per_process = n_total // size

    logging.debug(f"Working from rank {rank}")

    param_ranges = np.asarray([As, ns, taus, omega_bs, omega_cs, H0s])
    np.random.seed(1000 + rank)
    # get parameters
    parameters = np.random.uniform(param_ranges[:, 0], param_ranges[:, 1], (n_per_process, 6))
    # calculate spectra
    camb_cls = np.apply_along_axis(CAMBPowerspectrum, axis=1, arr=parameters, lmax=lmax)
    # returned in rank order, still matching up
    parameters = comm.gather(parameters, root=0)
    camb_cls = comm.gather(camb_cls, root=0)
    if rank == 0:
        # conactenate gathered arrays and save
        parameters = np.concatenate(parameters)
        camb_cls = np.concatenate(camb_cls)

        np.save(data_dir / "parameters.npy", parameters)
        np.save(data_dir / "spectra.npy", camb_cls)

    return

def main(argv):
    del argv

    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)

    # get config file path
    config_path = Path(FLAGS.config_file)

    # load configuration
    logging.debug(f"Using Gin config {config_path}")
    gin.parse_config_file(str(config_path))
    logging.debug(gin.config_str())

    
    # set data directory
    model_dir = MODEL_DIR / "datasets" / config_path.stem
    model_dir.mkdir(exist_ok=True, parents=True)

    CalculatePowerspectra(data_dir)

    # this has to go at the end, after gin has been used.
    with open(data_dir / "operative_gin_config.txt", "w") as f:
        f.write(gin.operative_config_str())

if __name__ == "__main__":
    flags.DEFINE_string('config_file', './configs/camb/planck_3sigma.gin', 'Configuration file for run.')
    flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
    app.run(main)
