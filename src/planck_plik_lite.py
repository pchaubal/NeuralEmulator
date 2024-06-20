import numpy as np
from dense import cv_loss
from datasets import NormalizeParams
import tensorflow as tf
import clik

# Take planck params
# Evaluate model on planck params
# feed cls to plik_lite along with Aplanck
# compare the value to corresponding value from planck chains
# calculate erros

tfkl = tf.keras.layers
tfk = tf.keras

def load_planckchain():
    filename = './../../clustercosmo/planck_chains/plikHM_TT_lowl_lowE/planck_TT_lowl_lowE.txt'
    nsamples = 100
    chain = np.loadtxt(filename)
    params = chain[:nsamples,[37,7,5,2,3,23]]
    params[:,0] *= 1e-9
    #Nuisance params
    nuisance = np.zeros((nsamples,0))
    nuisance = chain[:nsamples,8]
    # 
    lnL= chain[:nsamples,86]
    return params,nuisance, lnL

def Eval(params):
    params = NormalizeParams(params)
    model_dir = './models/Dense_128_tt/'
    model = tfk.models.load_model(model_dir, custom_objects={'cv_loss': cv_loss, 'cv_percent':cv_percent })
    predictions = 6000*model.predict(params)

    return predictions

def calculate_likelihood(lkl, Dls, nuisance):
    nsamples = len(Dls)
    l = np.arange(2,2552)
    cls = (2.*np.pi/(l*(l+1.)))[None,...] * Dls
    cl_and_pars = np.zeros((nsamples, 2510))
    cl_and_pars[:,2:2509] = cls[:,:2507]
    cl_and_pars[:,2509] = nuisance
      
#     lklfile ='/home/pc/codes/planck/plik_lite_v22_TT.clik/' 
#     lkl = clik.clik(lklfile)
    # Calculate likelihood
    lnL = lkl(cl_and_pars)
    print( lnL[:3] )

    return lnL

def errors(true, prediction):
    frac_err = (prediction - true) / true
    mean_err = np.mean(frac_err)
    err_on_err = np.std(frac_err)
    print( 'The error is ', mean_err, '+/-', err_on_err )
    return

def main():
    params, nui_par, chisq = load_planckchain()
    predicted_cls = Eval(params) # The last index is for Aplanck

    chisq_pred = calculate_likelihood(predicted_cls, nui_par)
    errors(chisq_pred,chisq)
    
    return

if __name__=='__main__':
    main()
