import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
import gin
from pathlib import Path
import datasets




@gin.configurable
def Eval(model, x_test, y_test, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    
    parameter_names = np.asarray(['log_As','ns','theta','ombh2','omch2','tau'])
    column_max = np.asarray([3.5, 1.1, 1.045, 0.025, 0.15, 0.15 ])
    column_min = np.asarray([2.5, .90, 1.035, 0.019, 0.10, 0.03 ])
    param_scaling = [ column_min, column_max, parameter_names ]
    scalingspace = datasets.ScaledSpace(param_scaling=param_scaling)

    predictions = scalingspace.MakePrediction(x_test, model)

    plot_reconstructions(y_test, predictions, results_dir)
    plot_frac_errors(y_test, predictions, results_dir)
    plot_abs_errors(y_test, predictions, results_dir)
    plot_cv_error(y_test, predictions, results_dir)
    plot_binned_errror(y_test, predictions, nbins=50, results_dir=results_dir)
#     plot_error_powerspectrum(y_test, predictions, results_dir)
    return


def plot_reconstructions(y_test, predictions, results_dir, seed=1234):
    np.random.seed(seed)

    fig, ax = plt.subplots(1, 1)
    for _ in range(3):
        idx = np.random.randint(y_test.shape[0])
        (l1,) = ax.plot(y_test[idx])
        ax.plot(predictions[idx], linestyle="--", color=l1.get_color())
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell$")
    ax.tick_params(axis="both", direction="inout")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(results_dir / "Reconstructions.pdf", bbox_inches="tight")
    return


def plot_frac_errors(y_test, predictions, results_dir):
    errors = predictions - y_test
    fractional_errors = errors / y_test

    l = np.arange(2, y_test.shape[1] + 2)

    fig, ax = plt.subplots(1, 1)
    for _ in range(3):
#         ax.plot(fractional_errors, color=l1.get_color(), alpha=0.3)
        ax.errorbar(l, fractional_errors.mean(axis=0), yerr = np.std(fractional_errors, axis=0), errorevery=20 )

    ax.axhline(0, linewidth=0.5, color="k")
    ax.set_ylim(-0.005, 0.005)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\Delta \mathcal{D}_\ell / \mathcal{D}_\ell $")
    ax.tick_params(axis="both", direction="inout")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(results_dir / "frac_errors.pdf", bbox_inches="tight")

    return

def plot_abs_errors(y_test, predictions, results_dir):
    errors = predictions - y_test
    err_mean = np.mean(errors, axis=0)
    err_sig = np.std(errors, axis=0)
    e_max = np.sort(errors, axis=0)[int(0.99*len(errors) -1)]
    e_min = np.sort(errors, axis=0)[int(0.01*len(errors))]

    l = np.arange(2, y_test.shape[1] + 2)

    fig, ax = plt.subplots(1, 1)
#     ax.errorbar(l, err_mean, yerr = err_sig, errorevery=20)
    ax.fill_between(l, y1 = err_mean+err_sig, y2 = err_mean-err_sig, zorder=3, alpha=0.8, label='1 $\sigma$' )
#     ax.fill_between(l, y1 = err_mean+3*err_sig, y2 = err_mean-3*err_sig, zorder=1, label='3 $\sigma$' )
    ax.fill_between(l, y1 = e_max, y2 = e_min, zorder=2, label='99 percentile' )
    ax.fill_between(l, y1 = np.max(errors,axis=0), y2 = np.min(errors,axis=0),alpha=0.8, zorder=1, label='max error' )
    ax.axhline(0, linewidth=0.5, color="k")

    planck_error = np.loadtxt('./src/bandpowers_planck.txt')
    ax.errorbar(planck_error[:,0], 
            np.zeros(planck_error.shape[0]), 
            yerr = planck_error[:,-2].T, 
            fmt = 'k',
            linewidth=0.1,
            elinewidth=0.1,
            label='Planck')

    ax.legend()
    ax.set_ylim(-10,10)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\Delta \mathcal{D}_\ell$")
    ax.tick_params(axis="both", direction="inout")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(results_dir / "abs_error.pdf", bbox_inches="tight")
    return

def plot_cv_error(y_test, predictions, results_dir):
    errors = predictions - y_test
    l = np.arange(2, y_test.shape[1] + 2)
    y_cv = np.sqrt(2.0 / (2 * l + 1)) * y_test
    cv_fraction = errors / y_cv
    cv_rms = np.sqrt(np.mean(cv_fraction ** 2, axis=0))
    cv_abs_err = np.sort(np.abs(cv_fraction), axis=0)

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        l,
        cv_abs_err[int(0.95 * len(cv_abs_err))],
        linestyle="--",
        label="95 percentile",
    )
    ax.plot(
        l,
        cv_abs_err[int(0.99 * len(cv_abs_err))],
        linestyle="--",
        label="99 percentile",
    )
    ax.plot(l, cv_rms, "k", label="RMS error")
    ax.errorbar(
        l,
        np.mean(cv_fraction, axis=0),
        yerr=np.std(cv_fraction, axis=0),
        elinewidth=0.1,
        errorevery=20,
        label="mean error",
    )
    [ax.axhline(y=i, linewidth=0.5, color="k", alpha=0.5) for i in [-0.01, 0, 0.01, 0.02]]
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\Delta D_\ell \slash CV$")
    ax.legend()
    ax.tick_params(axis="both", direction="inout")
    fig.savefig(results_dir / "CV_frac_error.pdf", bbox_inches="tight")

    return


def plot_binned_errror(y_test, predictions, nbins, results_dir):
    num_l = y_test.shape[1]
    r = num_l % nbins
    if r == 0:
        delta_l = num_l/ nbins
    else:
        delta_l = (num_l - r)/nbins
        y_test = y_test[:,:-r]
        predictions = predictions[:,:-r]
    lbin_edges = np.arange(2, num_l + 2, delta_l)
    l = 0.5*(lbin_edges[:-1] + lbin_edges[1:])
    y_test_binned = np.mean(y_test.reshape(len(y_test), nbins, -1), axis=2)
    predictions_binned = np.mean(
        predictions.reshape(len(predictions), nbins, -1), axis=2
    )
    binned_error = (predictions_binned - y_test_binned)
    binned_frac_error =  binned_error / y_test_binned
    binned_cv_error = binned_frac_error/np.sqrt(2 /( (2 * l + 1.0)* delta_l))  # cosmic variance factor
    binned_mean = np.mean(binned_error, axis=0)
    binned_error = np.std(binned_error, axis=0)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(
        l,
        binned_mean,
        yerr=binned_error,
        ls="--",
        capsize=3,
        capthick=0.5,
        elinewidth=0.5,
        ecolor="black",
    )
    ax.axhline(0, linewidth=0.5, color="k")
    # ax.set_ylim(-10.0, 10.0)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\Delta \mathcal{D}_{\ell, binned }$")
    fig.savefig(results_dir / "binned_error.pdf", bbox_inches="tight")

    return


def plot_error_powerspectrum(y_test, predictions, results_dir):
    errors = predictions - y_test
    l = np.arange(2, y_test.shape[1] + 2)
    y_cv = np.sqrt(2.0 / (2 * l + 1)) * y_test
    cv_fraction = errors / y_cv
    cvf_fft = np.apply_along_axis(np.fft.rfft, 1, cv_fraction)
    error_power = np.mean(cvf_fft * np.conj(cvf_fft), axis=0)

    k = np.fft.rfftfreq(2550, 1)
    fig, ax = plt.subplots(1, 1)
    ax.semilogx(k, error_power)
#     ax.errorbar(k,error_power,yerr=np.std(cvf_fft*np.conj(cvf_fft)),ls='--', capsize=3, capthick=0.5, elinewidth=0.5, ecolor='black')
    fig.savefig(results_dir / "error_power.jpeg", bbox_inches="tight", dpi=500)

    return

