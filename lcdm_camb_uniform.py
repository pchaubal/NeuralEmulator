mport numpy as np
from mpi4py import MPI
import camb
from camb import model, initialpower
from pathlib import Path


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def uniform_unit_sphere(ndim):
    """
    Taken from this math stack post:
    """
    x = np.random.randn(ndim)
    x_hat = x/np.linalg.norm(x)
    pt = x_hat*(np.random.rand())**(1/ndim)
    return pt

def unif_ellipsoid(cov):
    """
    Given a covmat, this throws points uniformly in the ellipse
    """
    ndim = cov.shape[0]
    a = np.zeros((ndim,ndim))
#     np.fill_diagonal(a,1e-15)
    cov += a
    cholesky = np.linalg.cholesky(cov)
    pt = np.dot(cholesky,uniform_unit_sphere(ndim))
    return pt
                                   

def get_powerspecta(params):
    pars.set_cosmology(thetastar=params[5]/100., ombh2=params[3], omch2=params[4], mnu=0.06, omk=0, tau=params[2])
    pars.InitPower.set_params(As=params[0], ns=params[1], r=0)
    pars.set_for_lmax(2600, lens_potential_accuracy=1)
    #Non-Linear spectra (Halofit)
    pars.set_matter_power(redshifts=[0.], kmax=10.0)
    pars.NonLinear = model.NonLinear_both
    
    results = camb.get_results(pars)
    powers =results.get_lensed_scalar_cls(CMB_unit='muK')
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    matter_pow = np.vstack((kh_nonlin,pk_nonlin[0]))

    return powers['total'], matter_pow


np.random.seed(int (rank+1234))

n_evals = 4000
# paramranges = np.asarray([ [1.8e-9, 3.0e-9], [.90,.99], [0.03,.25], [0.02, 0.024], [0.11,0.13], [50,95]])
mean = np.load('planck_mean_theta.npy')
cov = 16.*np.load('planck_covmat_theta.npy')

# pars = camb.CAMBparams()
pars = camb.read_ini("./planck_2018_params.ini")
parameters = np.zeros((n_evals,6))
cls = np.zeros((n_evals,2651,4))
pk = np.zeros((n_evals,2,200))
for i in range(n_evals):
#     params = np.random.uniform(paramranges[:,0], paramranges[:,1])
#     params = np.random.multivariate_normal(mean=mean, cov=cov)
    params = mean + unif_ellipsoid(cov) 
    cls[i], pk[i] = get_powerspecta(params) 
    parameters[i] = params
    if rank==0:
        print('Evaluations done on master:',i)

output_dir = Path('/data/projects/punim1108/camb_output/lcdm/uniform_theta')
output_dir.mkdir(exist_ok=True, parents=True)

fname = 'lcdm_cls_'+str(rank)+'.npy'
np.save(output_dir / fname, cls)

fname = 'lcdm_par_'+str(rank)+'.npy'
np.save(output_dir / fname, parameters)

fname = 'lcdm_pk_'+str(rank)+'.npy'
np.save(output_dir / fname, pk)
