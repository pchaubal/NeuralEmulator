import numpy as np
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

dest = '/home/pchaubal/ML/NeuralBoltzmann/results/dense_128_tt/'

def plot(samples,burnout=0, skip=1):
    names = ['$10^{10}A_s$','$ n_s $', '$ \\tau $', '$\Omega_bh^2$', '$\Omega_ch^2$','$\\theta^{*}$', '$A_{Planck}$']
#     chain = samples[burnout::1,:-1]
#     chain = samples[burnout::skip,[0,1,2,3,4,5,-2]]
    chain = samples[burnout::skip,:]
    mean = chain.mean(axis=0)
    planck_chain = np.loadtxt('/data/projects/punim1108/planck_chains/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE.txt')[:,[37,7,5,2,3,51,8]]
#     planck_chain = np.loadtxt('/data/projects/punim1108/planck_chains/base/plikHM_TT_lowl_lowE/base_plikHM_TT_lowl_lowE.txt')[:,[37,7,5,2,3,51, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,8]]
    c = ChainConsumer()
    c.add_chain(planck_chain, name='Planck TT')
#     c.add_chain(planck_chain, parameters=names, name='Planck TT ')
#     c.add_chain(chain,parameters=names, name='ML trained spectra')
    c.add_chain(chain, name='ML trained spectra')
#     c.configure(kde=False,smooth=0,bins=0.5,plot_color_params=True)
    c.configure(kde=False,bins=.5,plot_color_params=True)
#     truth=[2.1, .961, 0.051, 0.02212, 0.120, 66.8, 1.]
#     c.plotter.plot(filename="example.jpg", figsize="column", truth=truth)
    c.plotter.plot(filename= dest + "example.jpg", figsize="column" )
    plt.show()

if __name__=='__main__':
    samples = np.load('./planck_ml.npy')
#     samples[:,0] *= 1e9
    plot(samples, burnout=0, skip=1)
