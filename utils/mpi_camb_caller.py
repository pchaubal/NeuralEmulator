import numpy as np
from mpi4py import MPI
import camb
from camb import model, initialpower


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_powerspecta(params):
# pars = camb.CAMBparams()
    pars.set_cosmology(H0=params[5], ombh2=params[3], omch2=params[4], mnu=0.06, omk=0, tau=params[2])
    pars.InitPower.set_params(As=params[0], ns=params[1], r=0)
    pars.set_for_lmax(2600, lens_potential_accuracy=1)
    
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    return powers['total']


n_evals = 500
paramranges = np.asarray([ [2.0e-9, 2.2e-9], [.90,.99], [0.04,.08], [0.021, 0.023], [0.11,0.13], [50,95]])

pars = camb.CAMBparams()
parameters = np.zeros((n_evals,6))
cls = np.zeros((n_evals,2600,4))
for i in range(n_evals):
    params = np.random.uniform(paramranges[:,0], paramranges[:,1])
    cls[i] = get_powerspecta(params) 
    parameters[i] = params
    if rank==0:
        print('Evaluations done on proc 0:',i)

np.save('camb_cls_'+str(rank)+'.npy', cls)
np.save('camb_par_'+str(rank)+'.npy',parameters)
