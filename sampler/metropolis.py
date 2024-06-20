import numpy as np
import os
from scipy import stats

class MetropolisHastings():
    """Implements MH algorithm"""

    def __init__(self,likelihood,paramranges):
        self.posterior = None
        self.lnL = likelihood
        self.paramranges = paramranges
        self.ndims = paramranges.shape[0]
        # if previous file exists, delete it
        if os.path.isfile('samples.txt'):
            os.remove('samples.txt')
            print('Removed samples.txt from previous run')
        
    def MH(self,n_samples,initial_guess=None,cov=None,update_freq=100):
        self.update_freq = update_freq

        if initial_guess is None:
            initial_guess = np.mean(self.paramranges,axis=1)
        
        if cov is None:
            self.cov = np.identity(self.ndims)
        else:
            self.cov = cov
        
        lnL = self.lnL(initial_guess) 	#Initializing
        lnL += self.prior(initial_guess)
        print('lnl', lnL )
        old_point = initial_guess
        
        # Define an initial covmat
        self.cholesky()

        self.samples = initial_guess
        n_accepted = 0
        for i in range(n_samples):
            #Determine the new point
#             new_pt =  old_point + np.dot(proposal,np.random.randn(ndims))
            found_newpt = False
            while not found_newpt:
                new_pt = old_point + self.propose()
#                 found_newpt = self.is_within_boundaries(new_pt)
                if self.is_within_boundaries(new_pt):
                    lnL_new = self.lnL(new_pt) + self.prior(new_pt)
                    if not (lnL_new == -np.inf):
#                         print( lnL_new )
                        found_newpt = True
#             new_pt = old_point + self.propose()
#             while (np.any(new_pt>paramranges[:,1]) or np.any(new_pt<paramranges[:,0])): 
#                 new_pt =  old_point + np.dot(proposal,np.random.randn(ndims))

            lnL_new = self.lnL(new_pt) + self.prior(new_pt)
            

            if ( np.exp(lnL_new - lnL) > np.random.random() ):
                # Accept the new point
                self.samples = np.vstack((self.samples,new_pt))
#                 samples.append(new_pt)
                #Redefine the point
                old_point, lnL = new_pt, lnL_new
                # Update the number of accepted points
                n_accepted += 1
            else:
#                 samples.append(old_point)
                self.samples = np.vstack((self.samples,old_point))

            #Update the proposal matrix and print a feedback    
            if (i%self.update_freq==0):
                if i==0: continue # Dont do anything on first step
                self.acceptance = n_accepted/self.update_freq
                self.tune()
                self.update_covmat()
                n_accepted=0
                # write the data to a file
                with open('samples.txt','a') as f:
                    np.savetxt(f,self.samples)

                # Set samples to empty list as we have already saved these
                self.samples = self.samples[-1]
                # print a feedback
                print('\nLikelihood Evaluations:',i)
                print('lnL=',lnL)
                print('Acceptance ratio:', self.acceptance)

        return
    
    def prior(self,point):
        lnp  = 0. #np.log(stats.norm.pdf(point[0],5,1)) #+ np.log(stats.norm.pdf(point[1],3,2))
        return lnp

    def propose(self):
        # take random normal vector
        n = np.random.randn(self.ndims)
        # dot it with cholesky decomposed L to make an ellipse
        p = np.dot(self.L,n)
#         print( p )
        return p

    def tune(self):
        if (self.acceptance < .3):
            self.cov *= .99

        if (self.acceptance > .7):
            self.cov *= 1.01
        return

    def update_covmat(self):
#         self.cov += np.cov(self.samples.T)/self.update_freq 
        self.cov = np.cov(self.samples.T) 
        self.cholesky()
        return

    def cholesky(self):
        self.L = np.linalg.cholesky(self.cov)
        return

    def random_rotation(self):
        return

    def is_within_boundaries(self,pt):
        return( np.all(pt> self.paramranges[:,0]) and np.all(pt < self.paramranges[:,1]) )
