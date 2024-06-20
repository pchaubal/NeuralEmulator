import numpy as np
import tensorflow as tf

class plikpy:
    def __init__(self):
        nbins = 215 #bins for TT
        cov = np.load('./plik_data/c_matrix_plik_v22.npy')[:nbins,:nbins]
        #symmetrize the covmat
        for i in range(nbins):
            for j in range(i,nbins):
                cov[i,j] = cov[j,i]
        # find the inverse
        cinv = np.linalg.inv(cov)
        
        # Load planck bin weights
        bin_weight = np.zeros(2507)
        bin_weight[28:2507] = np.loadtxt('./plik_data/bweight.dat')[:2507-28]
        
        # Planck bin edges
        lbin_min = np.loadtxt('./plik_data/blmin.dat').astype(int) + 30
        lbin_max = np.loadtxt('./plik_data/blmax.dat').astype(int) + 30 +1 # +1 to convert fortran index to python index

        # Load Planck fiducial Cls
        cl_data = np.loadtxt('./plik_data/cl_cmb_plik_v22.dat') 
        self.bin_center = cl_data[:nbins,0]
        fid_cl = cl_data[:nbins,1]

        ell = np.arange(2,2509)
        
        # binning_matrix
        binning_matrix = np.zeros((2507,nbins))
        for i_bin in range(nbins): # loop over just 215 elements and is executed only once so didnt bother vectorizing
            binning_matrix[ lbin_min[i_bin]-2 : lbin_max[i_bin]-2 , i_bin ] = 1.0

        # finally cast everything into tensorflow variables
        self.cinv = tf.cast(cinv, tf.float32)
        self.bin_weight = tf.cast(bin_weight, tf.float32)
        self.lbin_min = tf.cast(lbin_min, tf.float32)
        self.lbin_max = tf.cast(lbin_max, tf.float32)
        self.binning_matrix = tf.cast(binning_matrix.T, tf.float32)
        self.fid_cl = tf.cast(fid_cl, tf.float32)
        self.ell = tf.cast(ell, tf.float32)


        return


    def loglik(self, Dls, A_planck):
        cl = (2.*np.pi/(self.ell*(self.ell+1.))) * Dls
        cl = tf.cast(cl, tf.float32)
        cl_binned = tf.linalg.matvec(self.binning_matrix, cl*self.bin_weight )
        cl_binned /= A_planck**2.
        d = self.fid_cl - cl_binned
#         lnL = -0.5*d.dot(self.cinv.dot(d))
        lnL = tf.reduce_sum(d*tf.linalg.matvec(self.cinv,d))/2.
        return lnL


if __name__=='__main__':
    #test for the spectra provided with Planck likelihood
    plik = plikpy()
    Dl_test = np.loadtxt('./plik_data/bf_lite_plikTTTEEE_v22b_lowl_simall.minimum.theory_cl')[:,1]
    A_planck = 1.
    lnL = plik.loglik(Dl_test, A_planck) 
    print( lnL )
