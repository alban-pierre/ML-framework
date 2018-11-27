import numpy as np

from utils.base import *
from policy import Policy
# from utils.redirect import Train_Test
from regressor.regressor import Gaussian_Average_Estimator
# from regressor.regressor import Gaussian_Average_KNN_Estimator
from regressor.regressor import gaussian_average_knn_estimator
from distance.lp_distance import *



class Random_Continuous_Policy(Policy):
    
    def __call__(self, mab):
        i_arm = []
        for (p,c,std,lg) in mab.space:
            if lg:
                r = np.exp(np.random.randn()*std + np.log(c))
            else:
                r = np.random.randn()*std + c
            i_arm.append(r)
        return tuple(i_arm)



class Random_Uniform_Continuous_Policy(Policy):
    
    def __call__(self, mab):
        i_arm = []
        le = len(mab.space)
        rd = np.random.randn(le)
        diaag = np.diag(-2*np.ones(le))+1
        diaag[0,0] = 1
        all_rd = [np.roll(d*s*rd, i) for i in range(le) for s in [1,-1] for d in diaag]
        for rd in all_rd:
            i_arm.append([])
            for i,(p,c,std,lg) in enumerate(mab.space):
                if lg:
                    # r = np.exp(rd[i]*std + np.log(c))
                    r = rd[i]*std + np.log(c)
                else:
                    r = rd[i]*std + c
                i_arm[-1].append(r)
        lstd = [std for (_,_,std,_) in mab.space]
        x = np.asarray(mab.array_rewards)
        if (x.shape[0] > 1):
            x = x[:,:-1]
            for i,(p,c,std,lg) in enumerate(mab.space):
                if lg:
                    x[:,i] = np.log(x[:,i])
            dists = asymmetric_distance_l2(x, np.asarray(i_arm), squared=True)
            i_a = np.argmax(np.min(dists, axis=0))
        else:
            i_a = 0
        r = i_arm[i_a]
        for i,(p,c,std,lg) in enumerate(mab.space):
            if lg:
                # r = np.exp(rd[i]*std + np.log(c))
                r[i] = np.exp(r[i]) #rd[i]*std + np.log(c)
        return tuple(r)



class Little_Less_Random_Continuous_Policy(Policy):
    
    def __call__(self, mab):
        i_arm = []
        coeffs = (1., 1.)
        mi = None
        if (len(mab.mean_rewards) > 2*len(mab.space)):
            mi = min(mab.mean_rewards, key=mab.mean_rewards.get)
            coeff = np.random.rand()*1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            if (np.random.rand() > coeff):
                if (np.random.rand() > coeff):
                    coeffs = (0., np.random.rand())
                else:
                    coeffs = (coeff, np.random.rand())
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                if lg:
                    cc = np.exp(np.log(c)*coeffs[0] + np.log(mi[i])*(1-coeffs[0]))
                else:
                    cc = c*coeffs[0] + mi[i]*(1-coeffs[0])
                sstd = std*coeffs[1]
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)



class Gaussian_UCB_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __init__(self, exploitation_over_exploration_ratio=None, knn_coeff=None):
        super(Gaussian_UCB_Continuous_Policy, self).__init__()
        self.exploitation_over_exploration_ratio = exploitation_over_exploration_ratio
        self.knn_coeff = knn_coeff

    def __call__(self, mab):
        i_arm = []
        if (self.exploitation_over_exploration_ratio is None):
            self.exploitation_over_exploration_ratio = 0.4
        if (self.knn_coeff is None):
            self.knn_coeff = 1.
        # gstd = 0.3
        mi = None
        if (len(mab.mean_rewards) > 10*len(mab.space)):
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            sh = x.shape[0]
            m = np.maximum(sh/4, 10)
            sq = np.maximum(int(self.knn_coeff*np.sqrt(sh)), 10)
            # gstd = np.maximum(2.,1.8+x.shape[0]/1000.)/x.shape[0]**0.25
            x_lg = np.zeros(x.shape)
            for i,(_,_,_,lg) in enumerate(mab.space):
                if lg:
                    x_lg[:,i] = np.log(x[:,i])
                else:
                    x_lg[:,i] = x[:,i]
            lstd = [std for (_,_,std,_) in mab.space]
            rs_std = np.random.randn(m, len(mab.space))*lstd
            if (x.shape[0] < 300):
                xc_ = [[(np.log(i[1]) if i[3] else i[1]) for i in mab.space]]
            else:
                xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
            x_lg_ = rs_std + xc_
            # g = Gaussian_Average_KNN_Estimator(Train_Test([(x_lg/lstd,y),(x_lg_/lstd,0)]), k=sq)
            # mi = g.test()
            g = Empty_Class()
            mi = gaussian_average_knn_estimator((x_lg/lstd,y), (x_lg_/lstd,0), k=sq, slf=g)
            mi = mi/np.std(mi)*0.3
            coef = self.exploitation_over_exploration_ratio
            weights = coef/np.mean(g.sorted_distance[:,:3]**2, axis=1)
            mi -= np.sqrt(2*np.log(sh)/weights)
            mi = tuple(x_lg_[np.argmin(mi),:])
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                if lg:
                    cc = np.exp(mi[i])
                else:
                    cc = mi[i]
                sstd = std*0.
            # self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)
