import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import LoadDataset, NormalizeParams
from pathlib import Path
from dense import cv_loss
from plot_eval import make_prediction
from planck_plik_lite import calculate_likelihood
import clik

tfk = tf.keras

def plot_1param_slice(model, param_scaling, results_dir):

    lklfile ='/home/pc/codes/planck/plik_lite_v22_TT.clik/' 
    lkl = clik.clik(lklfile)

    bestfit = np.asarray([2.093e-9, 0.9635, 0.052, 0.0221, 0.1206, 1.040])
#     param_min, param_max = param_scaling[0], param_scaling[1]
    n_samples = 100 # number of samples in each 1parameter slice
    
    pmin = np.asarray([2.08e-9, 0.955, 0.048, 0.0215, 0.118, 1.038])
    pmax = np.asarray([2.11e-9, 0.98, 0.055, 0.0225, 0.122, 1.042])


    paramnames = [r"$A_s$", r"$n_s$", r"$\tau$", r"$\omega_b$", r"$\omega_c$", r"$\theta$"]
    make_param_plots = True 
    if make_param_plots:
        for i_param in range(6):
            param_arr = np.tile(bestfit, (n_samples,1))
            sample_points = np.linspace(pmin[i_param], pmax[i_param], n_samples)
            param_arr[:, i_param] = sample_points

            print( i_param )
            spectra = make_prediction(param_arr, model)
            for n_samp,sp in enumerate(spectra):
                plt.plot(sp, linewidth=0.5,label=paramnames[i_param] + '=' + str(param_arr[n_samp,i_param]))
            plt.xlabel('$\ell$')
            plt.ylabel('$C_{\ell}$')
            plt.title(paramnames[i_param])
#             plt.legend()
            name = "spectra" + str(i_param) + ".pdf"
            plt.savefig(results_dir / name)
            plt.clf()

    # ------------- #
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), constrained_layout=True)
    for i_param,ax in zip(range(6),axes.flat):
        # Take some 100 points in the parameter range and keeping all other params fixed at bestfit
        # constant evaluate the likelihood
#         pmin = param_min[i_param]
#         pmax = param_max[i_param]
        param_arr = np.tile(bestfit, (n_samples,1))
        sample_points = np.linspace(pmin[i_param], pmax[i_param], n_samples)
        print( sample_points.shape )
        param_arr[:, i_param] = sample_points


#         param_arr, _ = NormalizeParams(param_arr, param_scaling)

#         spectra = 6000*model.predict(param_arr)
        spectra = make_prediction(param_arr, model)
        lnL = calculate_likelihood(lkl,spectra,1)

        peak_loc = sample_points[np.where(lnL==lnL.max())]
        print( peak_loc )

#         ax = fig.add_subplot(2, 3, i_param+1)
        ax.semilogy(sample_points, np.exp(lnL) )
        ax.axvline(peak_loc, linestyle='--', linewidth=0.5, color='r')
        ax.axvline(bestfit[i_param], linestyle='-', linewidth=0.5, color='k')
        ax.tick_params(axis="both", direction="inout")
        ax.set_xlabel(paramnames[i_param])
        ax.set_ylabel("$\\mathcal{L}$")
    fig.savefig(results_dir / "param_slices.pdf")

    return

if __name__=="__main__":
    results_dir = Path('./results/dense_128_tt/')
    model_dir = Path('/home/pc/codes/NeuralBoltzmann/dense_128_tt/checkpoints/checkpoint')
    model = tfk.models.load_model(model_dir, custom_objects={'cv_loss': cv_loss })

    dset = LoadDataset()
    param_scaling = np.load('/home/pc/codes/NeuralBoltzmann/dense_128_tt/checkpoints/checkpoint/param_scaling.npy')

    plot_1param_slice(model, param_scaling, results_dir)
