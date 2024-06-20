import numpy as np
from pathlib import Path
from scipy.stats import norm
import clik
import os
import logging
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(filename='sampler.log', level=logging.DEBUG)

DATA_DIR = Path(os.environ["NEURALBOLTZMANN_DATA_DIR"])

class Planck():
    
    def __init__(self):
        lklfile ="/home/pchaubal/cosmosis/Planck_data/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik"
        # Initialize it
        self.lkl = clik.clik(lklfile)
        path = DATA_DIR / 'dense_128_tt' / "checkpoints" / 'checkpoint' 
        
        self.model = keras.models.load_model(path, custom_objects={'cv_loss': self.cv_loss})
        self.mean = np.load('/home/pchaubal/ML/planck_mean_theta.npy') 
        self.cov = np.load('/home/pchaubal/ML/planck_covmat_theta.npy')
        self.cinv = np.linalg.inv(self.cov) 
        self.planck_bestfit_spectra = np.loadtxt('./src/planck_2018_lensedCls.dat')[:2550,1]
    

    def cv_loss(cl_true,cl_pred):
        l = tf.cast(tf.range(2, 2 + 2550), tf.float32)
        cv = tf.sqrt(2 / (2 * l + 1)) * cl_true # this is cosmic variance
        d = tf.math.subtract(cl_true, cl_pred)
        cv_percent = d / cv
        return tf.reduce_sum(cv_percent ** 2)


    def NormalizeParams(self, params):
        column_max = np.asarray([3.0e-9, 1.1, 0.2, 0.025, 0.14, 1.042 ])
        column_min = np.asarray([1.5e-9, .90, 0.0, 0.019, 0.11, 1.038 ])

        scaled_0_1_params = (params - column_min) / (column_max - column_min)

        return scaled_0_1_params

    def Eval(self, params):
        params = self.NormalizeParams(params)
        params = params.reshape(1,-1)
        predictions = self.model.predict(params)[0]
        predictions += self.planck_bestfit_spectra

        return predictions

    
    def calculate_likelihood_plik_lite(self, Dl, nuisance):
        l = np.arange(2,2552)
        cls = (2.*np.pi/(l*(l+1.))) * Dl
        cl_and_pars = np.zeros(2510)
        cl_and_pars[2:2509] = cls[:2507]
        cl_and_pars[2509] = nuisance
      
        lklfile ='/home/pc/codes/planck/plik_lite_v22_TT.clik/' 
        # Calculate likelihood
        lnL = self.lkl(cl_and_pars)[0]

        return lnL

    def is_within_bounds(self, params):
        res = ( (params[-1] < 1.00) and  (params[-1] > 0.996) )
        return res

    def within_ellipsoid(self, pt):
        pt = pt - self.mean 
        res = ( np.matmul( pt.T, np.matmul(self.cinv, pt)) < 9.0  )
        return res

    def prior(self, params, nuisance):
        # tau prior
        lnp = -(params[2] - 0.0522)**2.0 / (2 * 0.016**2.)
        # ----------------
        if (np.isscalar(nuisance)):
            # this is plik_lite
            lnp += -(nuisance - 1.0)**2. / (2. * 0.0025**2)
            return lnp
        
        return lnp


    def plik_lite(self, params):
#         if not self.within_ellipsoid(params[:6]):
#             logging.warning('Out of the training region:',np.array2string(params))
#             return -np.inf
        
        # ------ This is odd, using for test
#         nuisance = np.sqrt( params[6] )
        nuisance = params[6]
        # -----------

        Dls = self.Eval(params[:6])
        lnL = self.calculate_likelihood_plik_lite(Dls, nuisance) + self.prior(params, nuisance)

        if (np.isnan(lnL)):
            return -np.inf

        return lnL 

