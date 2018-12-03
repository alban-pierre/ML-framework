import numpy as np

from utils.base import *
from policy import Policy
from regressor.regressor import Gaussian_Average_Estimator
from regressor.regressor import Gaussian_Average_KNN_Estimator



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
            dists = l2_dist(x, np.asarray(i_arm))
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





class Gaussian_Average_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __call__(self, mab):
        i_arm = []
        coeffs = (1., 1.)
        mi = None
        if (len(mab.mean_rewards) > 2*len(mab.space)):
            coeff = 1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            # x = mab.mean_rewards.keys()
            # y = [mab.mean_rewards[k] for k in x]
            # x = np.asarray(x)
            xx = [np.log(x[:,i]) if lg else x[:,i] for i,(_,_,_,lg) in enumerate(mab.space)]
            xx = np.stack(xx, axis=1)
            mi = Gaussian_Average_Estimator((xx,y), std=coeff).train()
            mi = tuple(x[np.argmin(mi),:])
            if (np.random.rand() > coeff):
                if (np.random.rand() > coeff):
                    coeffs = (0., coeff)
                else:
                    coeffs = (0., 1.)
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                cc = mi[i]
                # if lg:
                #     cc = np.exp(np.log(c)*coeffs[0] + np.log(mi[i])*(1-coeffs[0]))
                # else:
                #     cc = c*coeffs[0] + mi[i]*(1-coeffs[0])
                sstd = std*coeffs[1]
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)

def rs(std):
    return np.random.randn()*std

def ri(x, m):
    return np.random.choice(x.shape[0], m)

from utils.redirect import Train_Test
from regressor.regressor import l2_dist

class Gaussian_UCB_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __call__(self, mab):
        i_arm = []
        gstd = 0.3
        mi = None
        if (len(mab.mean_rewards) > 2*len(mab.space)):
            # coeff = 1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            m = 2*len(mab.space)
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            # # x = mab.mean_rewards.keys()
            # # y = [mab.mean_rewards[k] for k in x]
            # # x = np.asarray(x)
            x_ = [np.exp(rs(std)+np.log(x[ri(x,m),i])) if lg else rs(std)+x[ri(x,m),i] for i,(_,_,std,lg) in enumerate(mab.space)]
            x_ = np.stack(x_, axis=1)
            xc_ = [[i[1] for i in mab.space]]
            x = np.concatenate([x, x_, xc_], axis=0)
            # x = np.concatenate([x, xc_], axis=0)
            xx = [np.log(x[:,i])/std if lg else x[:,i]/std for i,(_,_,std,lg) in enumerate(mab.space)]
            xx = np.stack(xx, axis=1)
            g = Gaussian_Average_Estimator(Train_Test([(xx[:-m-1,:],y),(xx,0)]), std=gstd)
            # g = Gaussian_Average_Estimator(Train_Test([(xx[:-1,:],y),(xx,0)]), std=gstd)
            mi = g.test()
            mi = mi/np.std(mi)
            weights = g.sum_weights
            weights[-m-1:] += mab.repeat_min/2.
            mi -= np.sqrt(2*np.log(x.shape[0])/weights)
            mi += (l2_dist(xx, xx[-1:,:])[:,0]/3)**16
            mi = tuple(x[np.argmin(mi),:])
            # coeffs = (0., coeff)
            # if (np.random.rand() > coeff):
            #     if (np.random.rand() > coeff):
            #         coeffs = (0., coeff)
            #     else:
            #         coeffs = (0., 1.)
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                cc = mi[i]
                # if lg:
                #     cc = np.exp(np.log(c)*coeffs[0] + np.log(mi[i])*(1-coeffs[0]))
                # else:
                #     cc = c*coeffs[0] + mi[i]*(1-coeffs[0])
                sstd = std*gstd#coeffs[1]
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)






class Gaussian_UCB_2_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __call__(self, mab):
        i_arm = []
        gstd = 0.3
        mi = None
        if (len(mab.mean_rewards) > 10*len(mab.space)):
            # coeff = 1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            # m = 2*len(mab.space)
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            m = np.maximum(x.shape[0]/10, 2)
            gstd = 2./x.shape[0]**0.25
            x_lg = np.zeros(x.shape)
            for i,(_,_,_,lg) in enumerate(mab.space):
                if lg:
                    x_lg[:,i] = np.log(x[:,i])
                else:
                    x_lg[:,i] = x[:,i]
            rs_std = [2*rs(std) for (_,_,std,_) in mab.space]
            x_lg_ = x_lg[ri(x,m),:] + rs_std
            if (x.shape[0] < 50):
                xc_ = [[i[1] for i in mab.space]]
            else:
                xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
            x_all = np.concatenate([x_lg, x_lg_, xc_], axis=0)
            lstd = [std for (_,_,std,_) in mab.space]
            xx = x_all/lstd
            g = Gaussian_Average_Estimator(Train_Test([(xx[:-m-1,:],y),(xx,0)]), std=gstd)
            mi = g.test()
            # if (x.shape[0] < 200):
            #     mi = mi/np.std(mi)*0.3
            # else:
            #     mi = mi/np.std(mi)
            mi = mi/np.std(mi)*(1-gstd)
            # mi = mi/np.std(mi)*0.3
            weights = g.sum_weights
            weights[-m-1:] += mab.repeat_min/2.
            mi -= np.sqrt(2*np.log(x.shape[0])/weights)
            if (x.shape[0] < 100):
                mi += (l2_dist(xx, xx[-1:,:])[:,0]/5)**16
            else:
                mi += (l2_dist(xx, xx[-1:,:])[:,0]/4)**16
                # all_std = np.std(x_lg, axis=0)[np.newaxis,:]
                # mi += (l2_dist(x_all/all_std, xc_/all_std)[:,0]/3)**16
            mi = tuple(x_all[np.argmin(mi),:])
            # for i,(_,_,_,lg) in enumerate(mab.space):
            #     if lg:
            #         mi[i] = np.exp(mi[i])
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                if lg:
                    cc = np.exp(mi[i])
                else:
                    cc = mi[i]
                sstd = std*gstd
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)
            # coeffs = (0., coeff)
            # if (np.random.rand() > coeff):
            #     if (np.random.rand() > coeff):
            #         coeffs = (0., coeff)
            #     else:
            #         coeffs = (0., 1.)

            # x_ = [np.exp(2*rs(std)+np.log(x[ri(x,m),i])) if lg else 2*rs(std)+x[ri(x,m),i] for i,(_,_,std,lg) in enumerate(mab.space)]
            # x_ = np.stack(x_, axis=1)
            # if (x.shape[0] < 50):
            #     xc_ = [[i[1] for i in mab.space]]
            # else:
            #     xc_ = np.mean(x, axis=0)[np.newaxis,:]
            # x = np.concatenate([x, x_, xc_], axis=0)
            # # x = np.concatenate([x, xc_], axis=0)
            # xx = np.zeros(x.shape)
            # for i,(_,_,std,lg) in enumerate(mab.space):
            #     if lg:
            #         xx[:,i] = np.log(x[:,i])/std
            #     else:
            #         xx[:,i] = x[:,i]/std
            # # xx = [np.log(x[:,i])/std if lg else x[:,i]/std for i,(_,_,std,lg) in enumerate(mab.space)]
            # # xx = np.stack(xx, axis=1)
            # g = Gaussian_Average_Estimator(Train_Test([(xx[:-m-1,:],y),(xx,0)]), std=gstd)
            # # g = Gaussian_Average_Estimator(Train_Test([(xx[:-1,:],y),(xx,0)]), std=gstd)
            # mi = g.test()
            # mi = mi/np.std(mi)
            # weights = g.sum_weights
            # weights[-m-1:] += mab.repeat_min/3.
            # mi -= np.sqrt(2*np.log(x.shape[0])/weights)
            # if (x.shape[0] < 50):
            #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/5)**16
            # else:
            #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/5)**16
            # mi = tuple(x[np.argmin(mi),:])
            # # coeffs = (0., coeff)
            # # if (np.random.rand() > coeff):
            # #     if (np.random.rand() > coeff):
            # #         coeffs = (0., coeff)
            # #     else:
            # #         coeffs = (0., 1.)
        # for i,(p,c,std,lg) in enumerate(mab.space):
        #     cc = c
        #     sstd = std
        #     if (mi is not None):
        #         cc = mi[i]
        #         # if lg:
        #         #     cc = np.exp(np.log(c)*coeffs[0] + np.log(mi[i])*(1-coeffs[0]))
        #         # else:
        #         #     cc = c*coeffs[0] + mi[i]*(1-coeffs[0])
        #         sstd = std*gstd#coeffs[1]
        #     self.all_centers.append(np.log(cc))
        #     if lg:
        #         r = np.exp(np.random.randn()*sstd + np.log(cc))
        #     else:
        #         r = np.random.randn()*sstd + cc
        #     i_arm.append(r)
        # return tuple(i_arm)






class Gaussian_UCB_3_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __call__(self, mab):
        i_arm = []
        gstd = 0.3
        mi = None
        if (len(mab.mean_rewards) > 10*len(mab.space)):
            # coeff = 1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            # m = 2*len(mab.space)
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            sh = x.shape[0]
            m = np.maximum(x.shape[0]/10, 2)
            x2 = x.copy()
            gstd = 3./x.shape[0]**0.25
            for i in range(x2.shape[1]):
                np.random.shuffle(x2[:,i])
            x_lg = np.zeros((x.shape[0]*2, x.shape[1]))
            for i,(_,_,_,lg) in enumerate(mab.space):
                if lg:
                    x_lg[:,i] = np.concatenate([np.log(x[:,i]), np.log(x2[:,i])])
                else:
                    x_lg[:,i] = np.concatenate([x[:,i], x2[:,i]])
            rs_std = np.random.randn(len(mab.space))*[std for (_,_,std,_) in mab.space]
            x_lg_ = x_lg[ri(x,m),:] + rs_std
            if (x.shape[0] < 500):
                xc_ = [[(np.log(i[1]) if i[3] else i[1]) for i in mab.space]]
            else:
                xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
            x_all = np.concatenate([x_lg, x_lg_, xc_], axis=0)
            lstd = [std for (_,_,std,_) in mab.space]
            xx = x_all/lstd
            # g = Gaussian_Average_Estimator(Train_Test([(xx[:-m-1,:],y),(xx,0)]), std=gstd)
            g = Gaussian_Average_Estimator(Train_Test([(xx[:sh,:],y),(xx[sh:,:],0)]), std=gstd)
            mi = g.test()
            # if (x.shape[0] < 200):
            #     mi = mi/np.std(mi)*0.3
            # else:
            #     mi = mi/np.std(mi)
            #mi = mi/np.std(mi)*0.3#*(1-gstd)
            mi = mi/np.std(mi)
            weights = g.sum_weights
            # weights[-m-1:] += mab.repeat_min/2.
            d = np.maximum((l2_dist(xx[sh:,:], xx[-1:,:])[:,0]/3)**2,1)
            weights += d - 1
            weights *= d
            mi -= np.sqrt(2*np.log(x.shape[0])/weights)
            # if (x.shape[0] < 100):
            #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/5)**16
            # else:
            #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/4)**16
            #     # all_std = np.std(x_lg, axis=0)[np.newaxis,:]
            #     # mi += (l2_dist(x_all/all_std, xc_/all_std)[:,0]/3)**16
            mi = tuple(x_all[np.argmin(mi)+sh,:])
            # for i,(_,_,_,lg) in enumerate(mab.space):
            #     if lg:
            #         mi[i] = np.exp(mi[i])
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                if lg:
                    cc = np.exp(mi[i])
                else:
                    cc = mi[i]
                sstd = std*gstd*0.2
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)





class Gaussian_UCB_4_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __call__(self, mab):
        i_arm = []
        gstd = 0.3
        mi = None
        if (len(mab.mean_rewards) > 10*len(mab.space)):
            # coeff = 1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            # m = 2*len(mab.space)
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            sh = x.shape[0]
            m = np.maximum(x.shape[0], 100)#np.maximum(x.shape[0]/10, 2)
            # x2 = x.copy()
            gstd = np.maximum(2.,1.8+x.shape[0]/1000.)/x.shape[0]**0.25
            # for i in range(x2.shape[1]):
            #     np.random.shuffle(x2[:,i])
            # x_lg = np.zeros((x.shape[0]*2, x.shape[1]))
            x_lg = np.zeros(x.shape)
            for i,(_,_,_,lg) in enumerate(mab.space):
                if lg:
                    x_lg[:,i] = np.log(x[:,i])
                else:
                    x_lg[:,i] = x[:,i]
            lstd = [std for (_,_,std,_) in mab.space]
            rs_std = np.random.randn(m,len(mab.space))*2*lstd
            # x_lg_ = rs_std#x_lg[ri(x,m),:] + rs_std
            if (x.shape[0] < 500):
                xc_ = [[(np.log(i[1]) if i[3] else i[1]) for i in mab.space]]
            else:
                xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
            x_lg_ = rs_std + xc_#x_lg[ri(x,m),:] + rs_std
            x_all = np.concatenate([x_lg, x_lg_, xc_], axis=0)
            xx = x_all/lstd
            # g = Gaussian_Average_Estimator(Train_Test([(xx[:-m-1,:],y),(xx,0)]), std=gstd)
            g = Gaussian_Average_Estimator(Train_Test([(xx[:sh,:],y),(xx[sh:,:],0)]), std=gstd)
            mi = g.test()
            # if (x.shape[0] < 200):
            #     mi = mi/np.std(mi)*0.3
            # else:
            #     mi = mi/np.std(mi)
            #mi = mi/np.std(mi)*0.3#*(1-gstd)
            mi = mi/np.std(mi)*0.3
            weights = g.sum_weights
            # weights[-m-1:] += mab.repeat_min/2.
            d = np.maximum((l2_dist(xx[sh:,:], xx[-1:,:])[:,0]/3)**2,1)
            weights += d - 1
            weights *= d
            mi -= np.sqrt(2*np.log(x.shape[0])/weights)
            # if (x.shape[0] < 100):
            #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/5)**16
            # else:
            #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/4)**16
            #     # all_std = np.std(x_lg, axis=0)[np.newaxis,:]
            #     # mi += (l2_dist(x_all/all_std, xc_/all_std)[:,0]/3)**16
            mi = tuple(x_all[np.argmin(mi)+sh,:])
            # for i,(_,_,_,lg) in enumerate(mab.space):
            #     if lg:
            #         mi[i] = np.exp(mi[i])
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                if lg:
                    cc = np.exp(mi[i])
                else:
                    cc = mi[i]
                sstd = std*gstd*0.2
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)




class Gaussian_UCB_5_Continuous_Policy(Policy):
    """
    must have array_rewards
    """
    def __init__(self, exploitation_over_exploration_ratio=None):
        super(Gaussian_UCB_5_Continuous_Policy, self).__init__()
        self.exploitation_over_exploration_ratio = exploitation_over_exploration_ratio

    def __call__(self, mab):
        i_arm = []
        if (self.exploitation_over_exploration_ratio is None):
            self.exploitation_over_exploration_ratio = 0.5
        gstd = 0.3
        mi = None
        if (len(mab.mean_rewards) > 10*len(mab.space)):
            # coeff = 1/np.sqrt(len(mab.mean_rewards)/(2*len(mab.space)))
            # m = 2*len(mab.space)
            x = np.asarray(mab.array_rewards)[:,:-1]
            y = np.asarray(mab.array_rewards)[:,-1]
            sh = x.shape[0]
            m = np.maximum(sh/4, 10)#np.maximum(x.shape[0]/10, 2)
            sq = np.maximum(int(2*np.sqrt(sh)), 10)
            # x2 = x.copy()
            gstd = np.maximum(2.,1.8+x.shape[0]/1000.)/x.shape[0]**0.25
            # for i in range(x2.shape[1]):
            #     np.random.shuffle(x2[:,i])
            # x_lg = np.zeros((x.shape[0]*2, x.shape[1]))
            x_lg = np.zeros(x.shape)
            for i,(_,_,_,lg) in enumerate(mab.space):
                if lg:
                    x_lg[:,i] = np.log(x[:,i])
                else:
                    x_lg[:,i] = x[:,i]
            lstd = [std for (_,_,std,_) in mab.space]
            rs_std = np.random.randn(m, len(mab.space))*lstd
            # x_lg_ = rs_std#x_lg[ri(x,m),:] + rs_std
            if (x.shape[0] < 300):
                xc_ = [[(np.log(i[1]) if i[3] else i[1]) for i in mab.space]]
            else:
                xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
            x_lg_ = rs_std + xc_#x_lg[ri(x,m),:] + rs_std
            g = Gaussian_Average_KNN_Estimator(Train_Test([(x_lg/lstd,y),(x_lg_/lstd,0)]), k=sq)
            mi = g.test()
            mi = mi/np.std(mi)*0.3
            coef = self.exploitation_over_exploration_ratio
            weights = coef/np.mean(g.sorted_distance[:,:3]**2, axis=1)
            mi -= np.sqrt(2*np.log(sh)/weights)
            mi = tuple(x_lg_[np.argmin(mi),:])
            #
            # x_all = np.concatenate([x_lg, x_lg_, xc_], axis=0)
            # xx = x_all/lstd
            # # g = Gaussian_Average_Estimator(Train_Test([(xx[:-m-1,:],y),(xx,0)]), std=gstd)
            # g = Gaussian_Average_Estimator(Train_Test([(xx[:sh,:],y),(xx[sh:,:],0)]), std=gstd)
            # mi = g.test()
            # # if (x.shape[0] < 200):
            # #     mi = mi/np.std(mi)*0.3
            # # else:
            # #     mi = mi/np.std(mi)
            # #mi = mi/np.std(mi)*0.3#*(1-gstd)
            # mi = mi/np.std(mi)*0.3
            # weights = g.sum_weights
            # # weights[-m-1:] += mab.repeat_min/2.
            # d = np.maximum((l2_dist(xx[sh:,:], xx[-1:,:])[:,0]/3)**2,1)
            # weights += d - 1
            # weights *= d
            # mi -= np.sqrt(2*np.log(x.shape[0])/weights)
            # # if (x.shape[0] < 100):
            # #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/5)**16
            # # else:
            # #     mi += (l2_dist(xx, xx[-1:,:])[:,0]/4)**16
            # #     # all_std = np.std(x_lg, axis=0)[np.newaxis,:]
            # #     # mi += (l2_dist(x_all/all_std, xc_/all_std)[:,0]/3)**16
            # mi = tuple(x_all[np.argmin(mi)+sh,:])
            # # for i,(_,_,_,lg) in enumerate(mab.space):
            # #     if lg:
            # #         mi[i] = np.exp(mi[i])
        for i,(p,c,std,lg) in enumerate(mab.space):
            cc = c
            sstd = std
            if (mi is not None):
                if lg:
                    cc = np.exp(mi[i])
                else:
                    cc = mi[i]
                sstd = std*gstd*0.
            self.all_centers.append(np.log(cc))
            if lg:
                r = np.exp(np.random.randn()*sstd + np.log(cc))
            else:
                r = np.random.randn()*sstd + cc
            i_arm.append(r)
        return tuple(i_arm)
