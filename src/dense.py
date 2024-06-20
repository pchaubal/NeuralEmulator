import os
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
from scipy.interpolate import CubicSpline
import tensorflow as tf
# import tensorflow_datasets as tfds
# from kerastuner.tuners import BayesianOptimization
# from num2tex import num2tex
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import cosmoplotian.colormaps
# string_cmap = "div yel grn"
# cmap = mpl.cm.get_cmap(string_cmap)
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[cmap(0.2), "k", "red"]) 
import datasets
from plot_eval import Eval

plt.rcParams['text.usetex'] = True
tfkl = tf.keras.layers
tfk = tf.keras
FLAGS = flags.FLAGS

DATA_DIR = Path(os.environ["NeuralBoltzmann_DATA"])
MODEL_DIR = Path(os.environ["NeuralBoltzmann_MODEL"])

@gin.configurable
def BuildDenseNetwork(input_dim, output_dim, units, activation='relu', dropout_rate=0.5):
    inputs = tfkl.Input(input_dim)
    x_ = tfkl.Dense(2 * input_dim)(inputs)
    for num_units in units:
        x_ = tfkl.Dense(num_units, activation=activation)(x_)
    x_ = tfkl.Dropout(dropout_rate)(x_)
    x_ = tfkl.Dense(output_dim, activation=activation)(x_)
    return tfk.Model(inputs=inputs, outputs=x_)

def cv_loss_func(cl_true,cl_pred,cv_factor):
    cv = tf.cast(cv_factor,tf.float32)
    d = tf.math.subtract(cl_true, cl_pred)
    cv_percent = d / cv
    return tf.reduce_sum(cv_percent ** 2)
    
def real_cv_loss(cv_factor):
    def local_cv_loss(y_true,y_pred):
        return cv_loss_func(y_true,y_pred,cv_factor)
    return local_cv_loss



@gin.configurable
def Train(model, x_train, y_train, val_data=None, batch_size=100, epochs=20, learning_rate=1e-3, loss=tfk.losses.MSE,cv_factor=None):
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate )
    if cv_factor is not None:
        model.compile(optimizer=optimizer, loss=real_cv_loss(cv_factor))
    else:
        model.compile(optimizer=optimizer, loss='mse')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=val_data)
    return model



def cv_loss(cl_true,cl_pred):
    l = tf.cast(tf.range(2, 2 + 2550), tf.float32)
    cv = tf.sqrt(2 / (2 * l + 1)) * cl_true # this is cosmic variance
    d = tf.math.subtract(cl_true, cl_pred)
    cv_percent = d / cv
    return tf.reduce_sum(cv_percent ** 2)


def CAMBSplineGradient(fiducial_x, gradient_idx, spline_anchors=[0.9, 0.95, 1. ,1.05, 1.1]):
    """ Function to compute the gradient of the power spectrum for a given
    fiducial set of parameters. Calculates the derivative with respect
    to the parameter at positino `gradient_idx`.
    """
    fiducial_value = fiducial_x[gradient_idx]
    spline_anchors = np.array([fiducial_value * frac for frac in spline_anchors])
    cl_anchors = []
    # Calculate 90% 95%, 105% and 110% of fiducial value to use a spline anchors.
    # Method used in 1509.06770 and seems robust.
    logging.debug(f"Computing spline anchors for parameter at index {gradient_idx}")
    for anchor in spline_anchors:
        logging.debug(f"Working on value {anchor:.02f}")
        pars = np.copy(fiducial_x)
        pars[gradient_idx] = anchor
        cl_anchors.append(datasets.CAMBPowerspectrum(pars))
    cl_anchors = np.concatenate([np.moveaxis(cl_anchor, -1, 0)[None, 0, :] for cl_anchor in cl_anchors])
    splineob = CubicSpline(spline_anchors, cl_anchors, axis=0)
    # compute spline derivative at fiducial value
    return splineob.derivative(nu=1)(fiducial_value)


def TFGradients(model, x_test, rawxtest, data_dir, idx=2000):
    """ Small initial exploration of gradients calculated from trained model.
    """
    logging.debug("Running gradients test ...")
    x_0 = rawxtest[idx] 
    x_0_tensor = tf.constant(x_test[[idx], :]) # get normalized version to use with NN
    logging.debug("Doing NN computation of gradients.")
    with tf.GradientTape() as tape:
        tape.watch(x_0_tensor)
        spectra = model(x_0_tensor)
    grads_tf = tape.batch_jacobian(spectra, x_0_tensor)
    # rescale things back to uK over the input range, and rescale param dependence
    param_ranges = np.max(rawxtest, axis=0) - np.min(rawxtest, axis=0)
    spectra_scale = 6000.
    rescaled_tf_grads = grads_tf.numpy()[0] / param_ranges[None, :] * spectra_scale
    # save both to file for quick plotting.grads_tf[0] * param_scales[None, :]
    np.save(data_dir / "tf_grads.npy", rescaled_tf_grads)


def SplineGradients(rawxtest, data_dir, idx=2000):
    x_0 = rawxtest[idx] # get unnormalied version of parameters to use with CAMB
    spline_grads = np.zeros((2501, 6))
    for i in range(6):
        spline_grads[:, i] = CAMBSplineGradient(x_0, i)
    logging.debug("Finished spline calculation.")
    np.save(data_dir / "spline_grads.npy", spline_grads)


def GradientsTestPlot(data_dir, results_dir, raw_x_test, idx=2000):
    tf_grads = np.load(data_dir / "tf_grads.npy")
    spline_grads = np.load(data_dir / "spline_grads.npy")
    parameters = raw_x_test[idx] # get parameters of fiducial model

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    axes[0, 0].plot(tf_grads[:, 0], "C0")
    axes[0, 0].plot(spline_grads[:, 0], "C0", linestyle="--")
    
    axes[0, 1].plot(tf_grads[:, 1], "C0")
    axes[0, 1].plot(spline_grads[:, 1], "C0", linestyle="--")

    axes[0, 2].plot(tf_grads[:, 2], "C0", label=r"${\rm NN~with~AD}$")
    axes[0, 2].plot(spline_grads[:, 2], "C0", linestyle="--", label=r"${\rm Numerical~via~spline}$")
    axes[0, 2].legend(frameon=False, loc='lower right', bbox_to_anchor=(1., 1.))
    
    axes[1, 0].plot(tf_grads[:, 3], "C0")
    axes[1, 0].plot(spline_grads[:, 3], "C0", linestyle="--")
    
    axes[1, 1].plot(tf_grads[:, 4], "C0")
    axes[1, 1].plot(spline_grads[:, 4], "C0", linestyle="--")
    
    axes[1, 2].plot(tf_grads[:, 5], "C0")
    axes[1, 2].plot(spline_grads[:, 5], "C0", linestyle="--")

    par_latex_list = [r"$A_s$", r"$n_s$", r"$\tau$", r"$\omega_b$",  r"$\omega_c$", r"$H_0$"]

    for X, ax in zip(par_latex_list, axes.flatten()):
        ax.set_ylabel(r"$\partial C_\ell / \partial$" + X + r"$~({\rm \mu K}^2)$")
        ax.tick_params(axis="both", direction="inout")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"$\ell$")

    textstr = r'$,~$'.join(
        [par + r"$={:.3g}$".format(num2tex(val)) for par, val in zip(par_latex_list, parameters)]
    )

    fig.suptitle(textstr)

    fig.savefig(results_dir / "gradients_test.pdf", bbox_inches="tight")
    return

def HyperparameterModelBuilder(hp):
    """ Function that is aware of hyperparameter choices, used with 
    keras-tuner to tune hyperparameters.
    """
    input_dim = 6
    output_dim = 2500
    scaling = 'linear'
    ntrain = 70_000
    label = 'prakrut'
    units = []
    for i in range(hp.Int('num_dense_layers', min_value=2, max_value=5)):
        units.append(hp.Choice(f'units_{i}', values=[128, 256, 512, 1024, 2048, 4096]))
    dropout = hp.Choice('dropout', values=[0.1, 0.5, 0.9])
    model = BuildDenseNetwork(input_dim, output_dim, units, activation='relu', dropout_rate=dropout)
    optimizer = tf.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]))
    model.compile(optimizer=optimizer, loss=tfk.losses.MSE)
    return model


def main(argv):
    del argv
    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
        tf.get_logger().setLevel('DEBUG')
    else:
        logging.set_verbosity(logging.INFO)
        tf.get_logger().setLevel('INFO')
    # get config file path
    config_path = Path(FLAGS.config_file)
    # use file name as identifier for saved files
    stem = config_path.stem
    results_dir = Path(FLAGS.results_dir) / stem
    results_dir.mkdir(exist_ok=True, parents=True)

    # set data directory
    model_dir = MODEL_DIR / stem
    model_dir.mkdir(exist_ok=True, parents=True)

    # parse configuration and lock it in
    logging.debug(f"Using Gin config {config_path}")
    gin.parse_config_file(str(config_path))
    logging.debug(gin.config_str())

    # setup checkpointing
    checkpoint_filepath = MODEL_DIR / stem / "checkpoints" / "checkpoint"
    checkpoint_filepath.parent.mkdir(exist_ok=True, parents=True)

    # setup hyperparameter search directory
    hyper_filepath = MODEL_DIR / stem / "hyperparameters" / "bayesianoptimization"
    hyper_filepath.parent.mkdir(exist_ok=True, parents=True)

    # The following is intended to be structured in a way that we do not need
    # to keep retraining networks, or recomputing large sets of spectra, during
    # the development of new applications. 
    # Keep training of NNs to the 'train' step, or additional new step at the beginning.
    # Save and reload for next applications.
    
    # These datasets are used in many different tasks so just unpack here.
    dset = datasets.LoadDataset()
    (x_train, y_train) = dset["train"]
    (x_val, y_val) = dset["val"]
    (x_test, y_test) = dset["test"]
    (raw_x_test, raw_y_test) = dset["raw_test"]

    ell = np.arange(2,dset["scalingspace"].lmax+1)
    cv_factor =  dset["scalingspace"].fiducial_spectrum * np.sqrt(2./(2*ell+1))
    # do training of NN
    if FLAGS.mode == "train":
        model = BuildDenseNetwork()
        model = Train(model, x_train, y_train, val_data=dset["test"],cv_factor=cv_factor)
        model.save(str(checkpoint_filepath))
        Eval(model, raw_x_test, raw_y_test, results_dir)

    if FLAGS.mode == "hyper_search":
        logging.debug("Beginning Hypersearch")
        tuner = BayesianOptimization(
            HyperparameterModelBuilder,
            objective='val_loss',
            max_trials=100,
            num_initial_points=10,
            executions_per_trial=3,
            directory=str(hyper_filepath),
            project_name='Boltzbot'
            )
        tuner.search_space_summary()
        tuner.search(x_train, y_train, epochs=20, verbose=0, validation_data=dset["test"])
        models = tuner.get_best_models(num_models=2)
        tuner.results_summary()

    # Calculate gradient for random test set parameters using trained NN
    if FLAGS.mode == "tf_gradients":
        model = tfk.models.load_model(str(checkpoint_filepath))
        TFGradients(model, x_test, raw_x_test, DATA_DIR)

    # Calculate gradient for random test set parameters using spline interpolation
    if FLAGS.mode == "spline_gradients":
        SplineGradients(raw_x_test, DATA_DIR)

    # Plot comparison of two gradient calculations above
    if FLAGS.mode == "plot_gradients":
        GradientsTestPlot(DATA_DIR, results_dir, raw_x_test)
    
    # Load a previously defined model and make plots 
    if FLAGS.mode =='eval':
        model = tfk.models.load_model(str(checkpoint_filepath))
        Eval(model, x_val, y_val, results_dir)

    with open(results_dir / "operative_gin_config.txt", "w") as f:
        f.write(gin.operative_config_str())
    return

if __name__ == "__main__":
    flags.DEFINE_enum("mode", "train", ["train", "eval", "tf_gradients", "spline_gradients", "plot_gradients", "hyper_search"], "Mode in which to run script.")
    flags.DEFINE_string("results_dir", "./results", "Path to results directory where plots will be saved.")
    flags.DEFINE_string("config_file", "./configs/dense_128_tt.gin", "Path to configuration file.")
    flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
    app.run(main)
