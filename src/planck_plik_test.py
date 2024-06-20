import numpy as np
from chainconsumer import ChainConsumer
import scipy
import matplotlib.pyplot as plt
from dense import cv_loss
from datasets import CAMBPowerspectrum
import datasets
import tensorflow as tf
import clik
import camb
from camb import model, initialpower


pars = camb.CAMBparams()
# Take planck params
# Evaluate model on planck params
# feed cls to plik_lite along with Aplanck
# compare the value to corresponding value from planck chains
# calculate errors

tfkl = tf.keras.layers
tfk = tf.keras

def load_planckchain():
    nsamples = 10000
    filename = '/home/pc/codes/clustercosmo/planck_chains/plikHM_TT_lowl_lowE/planck_TT_lowl_lowE.txt'

    chain = np.loadtxt(filename)
    params = chain[:nsamples,[37,7,5,2,3,51]]
    params[:,0] *= 1e-9
    #Nuisance params
    nuisance = np.zeros((nsamples,20))
    nuisance[:,0] = chain[:nsamples,9]
    nuisance[:,1] = -1.3
    nuisance[:,2:13] = chain[:nsamples, 10:21]
    nuisance[:,13:17] = 1.
    nuisance[:,17] = chain[:nsamples,21]
    nuisance[:,18] = chain[:nsamples,22]
    nuisance[:,19] = chain[:nsamples,8]
    # 
    lnL_plik = -0.5* chain[:nsamples,86] 

    return params,nuisance, lnL_plik

def Eval(params):
    model_dir = './dense_128_tt/checkpoints/checkpoint/'
    model = tfk.models.load_model(model_dir, compile=False )
    print( 'running ml model', params.shape )
    scalingspace = datasets.ScaledSpace()
    predictions = scalingspace.MakePrediction(params, model)
    return predictions

def calculate_likelihood(Dls, nuisance):
    nsamples = len(Dls)
    l = np.arange(2,2552)
    cls = (2.*np.pi/(l*(l+1.)))[None,...] * Dls
    cl_and_nui = np.zeros((nsamples, 2529))
    cl_and_nui[:,2:2509] = cls[:,:2507]
    cl_and_nui[:,2509:] = nuisance
     
    lklfile ='/home/pc/codes/planck/plik_rd12_HM_v22_TT.clik/' 
    # Initialize it
    lkl = clik.clik(lklfile)

    lnL = np.zeros(nsamples)
    for i in range(nsamples):
        lnL[i] = lkl(cl_and_nui[i])
#         print( i, lnL[i])

    return lnL 

def errors(true, prediction):
#     print( prediction - true )
    err = (prediction - true)
    print( '5 max errors:', np.sort(np.abs(err))[-5:] )
    mean_err = np.mean(err)
    err_on_err = np.std(err)
    print( 'The error is ', mean_err, '+/-', err_on_err )
    return err

def resample_chain(lnL_true, lnL_pred):
    
    filename = '/home/pc/codes/clustercosmo/planck_chains/plikHM_TT_lowl_lowE/planck_TT_lowl_lowE.txt'
    chain = np.loadtxt(filename)[:len(lnL_true)]

    weight = chain[:,0]

    delta_lnL = lnL_true - lnL_pred
    new_wt = weight * np.exp(delta_lnL)
    
    print( weight )
    print( new_wt )
#     plt.hist(new_wt)
#     plt.hist(weight)
#     plt.show()

    # separate the integer and fractional parts of weight
    # round the fractional parts to 0 or 1 according the prob. = fraction itself
    frac_wt, int_wt = np.modf( new_wt )
    ind = np.where(frac_wt > np.random.rand(len(delta_lnL)))
    new_chain = np.repeat(chain, int_wt.astype(int), axis=0)    
    new_chain = np.vstack( (new_chain, chain[ind]) )

    params = new_chain[:,[37,7,5,2,3,51,8]]
    c = ChainConsumer()
    c.add_chain(params, name='Planck TT')
    c.plotter.plot(filename= "example.jpg", figsize="column" )
    plt.show()
    
    
    return new_chain

def debug(error, params, dl):
    idx = np.argsort(np.abs(error))[-3:]

    planck_cov = np.load('./src/plik_data/c_matrix_plik_v22.npy')[:215,:215]
    w, v = np.linalg.eigh(planck_cov)

    l = np.arange(2,2509)
    print( dl[42].shape )
    bins = np.concatenate((np.arange(30,100,5), np.arange(100,1504,9), np.arange(1504,2014,17), np.arange(2014, 2509, 33)))
    bins = np.append(bins,2508)
    #plot the diff between camb and predicted spectra
    for i in idx:
        cls = (2.*np.pi/(l*(l+1)))*dl[i,:2507]
        binned_dl,_,_ = scipy.stats.binned_statistic(l,cls, bins=bins)
        np.dot(binned_dl,v)
#         print( i )
#         print( params[i] )
#         camb_dl = CAMBPowerspectrum(params[i])[:2550,1]
#         print( camb_dl.shape )
#         plt.plot(camb_dl)
#         plt.plot(dl[i])
#         plt.plot((dl[i] - camb_dl))
#     plt.show()
    return

def main():
    print( 'Runnning ...' )
    params, nui_par, lnL_true = load_planckchain()
    print( 'Read Planck chains' )
    predicted_Dls = Eval(params) # The last index is for Aplanck
    print( 'generated ml predictions' )

    lnL_pred = calculate_likelihood(predicted_Dls, nui_par)
    print( 'calculated planck likelihood on ml spectra' )
    error = errors(lnL_pred,lnL_true)
    resample_chain(lnL_true,lnL_pred)
#     print( 'entering debug' )
#     debug(error, params, predicted_Dls)
    
    return

if __name__=='__main__':
    main()
