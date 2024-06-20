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

print( clik.__file__ )
exit()
pars = camb.CAMBparams()
tfkl = tf.keras.layers
tfk = tf.keras

class Likelihood_Test():
    
    def __init__(self, limit_samples = None):
        
        self.limit_samples = limit_samples # the number of samples from planck chain

        # Initialize plik 
        lklfile ='/home/pc/codes/planck/plik_rd12_HM_v22_TT.clik/' 
        self.lkl = clik.clik(lklfile)
        return


    def LoadPlanckchain(self, fname='/home/pc/codes/clustercosmo/planck_chains/plikHM_TT_lowl_lowE/planck_TT_lowl_lowE.txt'):

        chain = np.loadtxt(fname)

        params = chain[:,[37,7,5,2,3,51]]
        params[:,0] *= 1e-9
        self.params = params 

        #Nuisance params
        nuisance = np.zeros( (len(chain),20) )
        nuisance[:,0] = chain[:,9]
        nuisance[:,1] = -1.3
        nuisance[:,2:13] = chain[:,10:21]
        nuisance[:,13:17] = 1.
        nuisance[:,17] = chain[:,21]
        nuisance[:,18] = chain[:,22]
        nuisance[:,19] = chain[:,8]

        nuisance = nuisance

        # Chisq CMB only from planck chains
        lnL_planck = -0.5* chain[:,86] 

        if self.limit_samples is not None:
            params = params[:self.limit_samples,:]
            nuisance = nuisance[:self.limit_samples,:]
            lnL_planck = lnL_planck[:self.limit_samples]

        return params, nuisance, lnL_planck


    def MLpredict(self, params):
        model_dir = './dense_128_tt/checkpoints/checkpoint/'
        model = tfk.models.load_model(model_dir, compile=False )
        print( 'running ml model', params.shape )
        scalingspace = datasets.ScaledSpace()
        predictions = scalingspace.MakePrediction(params, model)
        return predictions


    def Evaluate_plik(self, Dls, nuisance):
        Dls = Dls[:,:2507]
        nsamp = len(Dls)

        assert( len(Dls)==len(nuisance) )

        l = np.arange(2,2509)


        # convert Dls to cls
        cls = ( 2.*np.pi/(l*(l+1.)) )[None,...] * Dls

        # make an array of cls and nuisance as required by plik
        cl_and_nui = np.zeros((nsamp, 2529))
        cl_and_nui[:,2:2509] = cls[:,:2507]
        cl_and_nui[:,2509:] = nuisance
        
        # lnL = np.zeros(nsamp)
        # for i in range(nsamp):
            # lnL[i] = self.lkl(cl_and_nui[i])
        lnL = self.lkl(cl_and_nui)
        return lnL


    def errors(self, lnL_true, lnL_pred, plot_name='lnL_error_hist.jpg'):
        err = lnL_pred - lnL_true
        abs_err = np.abs(err)

        max_5_errors = np.sort(abs_err)[-5:]

        plt.hist(err)
        dname ="./results/dense_128_tt/" 
        plt.savefig(dname + plot_name)
        return
   
    def compare_ml_to_planck(self):
        params, nuisance, lnL_planck = self.LoadPlanckchain()

        ml_spectra = self.MLpredict(params)
        lnL_ml = self.Evaluate_plik(ml_spectra, nuisance)

        self.errors(lnL_planck, lnL_ml, 'ml_planck_hist.jpg')
        return

    def compare_camb_to_ml(self):
        camb_params = np.load('/home/pc/codes/ML/camb_data/uniform_theta/params.npy')[:self.n_samples]
        camb_spectra = np.load('/home/pc/codes/ML/camb_data/uniform_theta/lcdmcl_tt_small.npy')[:self.n_samples, 2:2552]

        # same nuisance params as planck chains to enure nuisance params are in proper range. Although, quite a sensible choice
        _ , nuisance, _  = self.LoadPlanckchain()
        assert( len(camb_params)==len(camb_spectra))
        assert( len(spectra)==len(nuisance))
        lnL_camb = self.Evaluate_plik(spectra, nuisance)

        ml_spectra = self.MLpredict(params)
        lnL_ml = self.Evaluate_plik(ml_spectra, nuisance)
        
        self.errors(lnL_camb, lnL_ml,'camb_ml_hist.jpg')
        return

    def compare_camb_to_planck(self):
        params, nuisance, lnL_planck = self.LoadPlanckchain()
        
        camb_params = np.load('/home/pc/codes/ML/camb_data/planck_TT_lowl_lowE/planck_TT_lowl_lowE_par.npy')
        # camb_spectra = np.load('/home/pc/codes/ML/camb_data/planck_TT_lowl_lowE/planck_TT_lowl_lowE_cltt.npy')[:,2:2552]
        camb_spectra = np.fromfile('../ML/camb_data/cls_for_prakrut.bin', dtype=np.float32).reshape(-1,2507)
        
        if self.limit_samples is not None:
            camb_spectra = camb_spectra[:self.limit_samples,:]
            camb_params = camb_params[:self.limit_samples,:]

        assert( (params == camb_params).all )
        
        # print( lnL_planck[88] ) 
        # cl2 = np.loadtxt('/home/pc/Downloads/ro88/row88_lensedCls.dat')[:2550,1]
        # cl2 = np.loadtxt('./worst_cls.txt')
        # print( cl2.shape )
        # cl2 = np.loadtxt('/home/pc/Downloads/ro88/row88_lensedCls.dat')
        # print( cl2.shape )
        # cl2 = np.asarray([cl2.T])
        # lnl2 = self.Evaluate_plik(cl2,nuisance[88:89])
        # print( lnl2 )

        lnL_camb = self.Evaluate_plik(camb_spectra, nuisance)

        # print( lnL_planck - lnL_camb )
        print(lnL_planck )
        print( lnL_camb )
        # print(np.argmax( lnL_planck - lnL_camb ))
        # print(np.max( lnL_planck - lnL_camb ))
        exit()
        
        self.errors(lnL_planck, lnL_camb, 'camb_planck_hist.jpg')
        return
    

if __name__=='__main__':
#     lkl_test = Likelihood_Test(5000)
#     lkl_test.compare_ml_to_planck()
    # if you want to test the likelihood wrt to some spectra
#     lkl_test.compare_camb_to_planck()
    # ------
    # if you want to check your camb spectra to planck (requires specrta evaluated at camb params)
    lkl_test = Likelihood_Test(100)
    lkl_test.compare_camb_to_planck()

