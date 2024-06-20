import numpy as np
from scipy.stats import norm
import clik
import os
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Planck():
    
    def __init__(self):
        lklfile ='/home/pc/codes/planck/plik_lite_v22_TT.clik/' 
#         lklfile ='/home/pc/codes/planck/plik_rd12_HM_v22_TT.clik/' 
        # Initialize it
        self.lkl = clik.clik(lklfile)
#         self.pico = pypico.load_pico("../pypico/jcset_py3.dat")  
        self.model = keras.models.load_model('/home/pc/codes/NeuralBoltzmann/dense_128_tt/checkpoints/checkpoint/', custom_objects={'cv_loss': self.cv_loss})
#         self.param_scaling = np.load('/home/pc/codes/NeuralBoltzmann/dense_128_tt/checkpoints/checkpointparam_scaling.npy')
        self.mean = np.load('/home/pc/codes/ML/planck_mean_theta.npy') 
#         self.mean[0] *= 1e-9
        self.cov = np.load('/home/pc/codes/ML/planck_covmat_theta.npy')
        self.cinv = np.linalg.inv(self.cov) # 16 for 4 sigma region cutoff
        self.planck_bestfit = np.loadtxt('/home/pc/codes/NeuralBoltzmann/src/planck_2018_lensedCls.dat')[:2550,1]
    

    def cv_loss(cl_true,cl_pred):
        l = tf.cast(tf.range(2, 2 + 2550), tf.float32)
        cv = tf.sqrt(2 / (2 * l + 1)) * cl_true # this is cosmic variance
        d = tf.math.subtract(cl_true, cl_pred)
        cv_percent = d / cv
        return tf.reduce_sum(cv_percent ** 2)


    def NormalizeParams(self, params):
        column_max = np.asarray([3.0e-9, 1.1, 0.2, 0.025, 0.14, 1.042 ])
        column_min = np.asarray([1.5e-9, .90, 0.0, 0.019, 0.11, 1.038 ])
#         column_min = self.param_scaling[0]
#         column_max = self.param_scaling[1]
        scaled_0_1_params = (params - column_min) / (column_max - column_min)
        return scaled_0_1_params

    def Eval(self, params):
        params = self.NormalizeParams(params)
        params = params.reshape(1,-1)
        predictions = self.model.predict(params)[0]
        predictions += self.planck_bestfit

        return predictions

    def calculate_likelihood_plik(self, Dl, nuisance):
        l = np.arange(2,2552)
        cls = (2.*np.pi/(l*(l+1.))) * Dl
        cl_and_pars = np.zeros(2529)
        cl_and_pars[2:2509] = cls[:2507]
        cl_and_pars[2509:] = nuisance
      
        lklfile ='/home/pc/codes/planck/plik_rd12_HM_v22_TT.clik/' 
        # Initialize it
        lkl = clik.clik(lklfile)
        # Calculate likelihood
        lnL = self.lkl(cl_and_pars)[0]

        return lnL
    
    def calculate_likelihood_plik_lite(self, Dl, nuisance):
        l = np.arange(2,2552)
        cls = (2.*np.pi/(l*(l+1.))) * Dl
        cl_and_pars = np.zeros(2510)
        cl_and_pars[2:2509] = cls[:2507]
        cl_and_pars[2509] = nuisance
      
        lklfile ='/home/pc/codes/planck/plik_lite_v22_TT.clik/' 
        # Initialize it
#         lkl = clik.clik(lklfile)
        # Calculate likelihood
        lnL = self.lkl(cl_and_pars)[0]

        return lnL

    def is_within_bounds(self, params):
        res = ( (params[-1] < 1.00) and  (params[-1] > 0.996) )
        return res

    def within_ellipsoid(self, pt):
        pt = pt - self.mean 
        res = ( np.matmul( pt.T, np.matmul(self.cinv, pt)) < 16.0  )
        return res

    def prior(self, params, nuisance):
        # tau prior
        lnp = -(params[2] - 0.052)**2.0 / (2 * 0.032**2.)
        # ----------------
        if (np.isscalar(nuisance)):
            # this is plik_lite
#             lnp += -(nuisance - 1.0)**2. / (2. * 0.0025**2)
            return lnp
        
        # Planck nuisance params
        # 0. acib217	A^{CIB}_{217}
        # 1. cib_index = -1.3
        # 2. xi	\xi^{tSZ-CIB}
        # 3. asz143	A^{tSZ}_{143}
        # 4. aps100	A^{PS}_{100}
        # 5. aps143	A^{PS}_{143}
        # 6. aps143217	A^{PS}_{143\times 217}
        # 7. aps217	A^{PS}_{217}
        # 8. aksz	        A^{kSZ}
        # -----------------
        # dust residual contamination
        # taken from plik_recommended_priors
        # 9. kgal100    / gal545_A_100     = 8.6  ± 2
        lnp += -(nuisance[10] - 8.6)  / (2. * 2.0**2.)
        # 10. kgal143    / gal545_A_143     = 10.6 ± 2
        lnp += -(nuisance[11] - 10.6) / (2. * 2.0**2.)
        # 11. kgal143217 / gal545_A_143_217 = 23.5 ± 8.5 
        lnp += -(nuisance[12] - 23.5) / (2. * 8.5**2.)
        # 12. kgal217}   / gal545_A_217     = 91.9 ± 20
        lnp += -(nuisance[13] - 91.9) / (2. * 20.**2.)
        # ------------------
        # These 4 params are set to 1
        # 13. A_sbpx_100_100_TT
        # 14. A_sbpx_143_143_TT
        # 15. A_sbpx_143_217_TT
        # 16. A_sbpx_217_217_TT
        # ------------------
        # taken from plik_recommended_priors
        # 17. cal0	c_{100}
        lnp += -(nuisance[18] - 1.0002) / (2. * 0.0007**2.)
        # 18. cal2	c_{217}
        lnp += -(nuisance[19] - 0.99805) / (2. * 0.00065**2.)
        # 19. A_planck/calPlanck prior
        lnp += -(nuisance[19]- 1.0)**2.0/(2 * 0.0025**2.)
        # ------------------
        #
        # 
        return lnp

    def planck_lik(self, params):
        
#         if not self.within_ellipsoid(params[:6]):
#             print( 'Point outside the training region' )
#             try:
#             print( 'Running CAMB. This is slow. If happens often, retrain' )
#                 run_camb(params)
#             except:
                # return -inf to set a strict cutoff
#             return -np.inf
        
        # Nuisance parameters
        nuisance = np.zeros(20)
        nuisance[0] = params[6]
        nuisance[1] = -1.3
        nuisance[2:13] = params[7:18]
        nuisance[13:17] = 1.
        nuisance[17] = params[18]
        nuisance[18] = params[19]
        nuisance[19] = params[20]
        # -----------------

        Dls = self.Eval(params[:6])
        lnL = self.calculate_likelihood_plik(Dls, nuisance) + self.prior(params,nuisance)

        if (np.isnan(lnL)):
            return -np.inf

        return lnL
   

    def plik_lite(self, params):
        if not self.within_ellipsoid(params[:6]):
#             print( 'point out of training region' )
            return -np.inf

        nuisance = np.sqrt( params[6] )

        Dls = self.Eval(params[:6])
        lnL = self.calculate_likelihood_plik_lite(Dls, nuisance) + self.prior(params, nuisance)
        if (np.isnan(lnL)):
            print( 'point is nan' )
            return -np.inf

        return lnL 

