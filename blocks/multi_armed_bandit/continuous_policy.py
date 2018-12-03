import numpy as np

from utils.base import *
from policy import Policy
# from utils.redirect import Train_Test
# from regressor.regressor import Gaussian_Average_Estimator
# from regressor.regressor import Gaussian_Average_KNN_Estimator
from regressor.regressor import gaussian_average_knn_estimator
from regressor.regressor import gaussian_average_max_knn_estimator
from regressor.regressor import gaussian_average_median_knn_estimator
from distance.lp_distance import *



class Continuous_Policy(Policy):

    def estimator_train(self, points=None):
        return estimator_test(self, points)

    def estimator_test(self, points=None):
        if (points == None):
            points = self.mab.array_rewards[:,:-1]
        x = self.mab.array_rewards[:,:-1]
        y = self.mab.array_rewards[:,-1]
        return gaussian_average_knn_estimator((x,y), (points,None), k=10)



class Random_Continuous_Policy(Continuous_Policy):

    def __call__(self):
        i_arm = []
        for (p,c,std,lg) in self.mab.space:
            if lg:
                r = np.exp(np.random.randn()*std + np.log(c))
            else:
                r = np.random.randn()*std + c
            i_arm.append(r)
        return tuple(i_arm)



class Random_Uniform_Continuous_Policy(Continuous_Policy):
    
    def __call__(self):
        i_arm = []
        le = len(self.mab.space)
        rd = np.random.randn(le)
        diaag = np.diag(-2*np.ones(le))+1
        diaag[0,0] = 1
        all_rd = [np.roll(d*s*rd, i) for i in range(le) for s in [1,-1] for d in diaag]
        for rd in all_rd:
            i_arm.append([])
            for i,(_,c,std,lg) in enumerate(self.mab.space):
                if lg:
                    # r = np.exp(rd[i]*std + np.log(c))
                    r = rd[i]*std + np.log(c)
                else:
                    r = rd[i]*std + c
                i_arm[-1].append(r)
        lstd = [std for (_,_,std,_) in self.mab.space]
        # x = np.asarray(self.mab.array_rewards)
        x = self.mab.array_rewards
        if (x.shape[0] > 1):
            x = x[:,:-1].copy()
            for i,(_,_,std,lg) in enumerate(self.mab.space):
                if lg:
                    x[:,i] = np.log(x[:,i])/std
                else:
                    x[:,i] = x[:,i]/std
                dists = asymmetric_distance_l2(x, np.asarray(i_arm)/lstd, squared=True)
            i_a = np.argmax(np.min(dists, axis=0))
        else:
            i_a = 0
        r = i_arm[i_a]
        for i,(_,_,_,lg) in enumerate(self.mab.space):
            if lg:
                # r = np.exp(rd[i]*std + np.log(c))
                r[i] = np.exp(r[i]) #rd[i]*std + np.log(c)
        return tuple(r)



# class Little_Less_Random_Continuous_Policy(Policy):
    
#     def __call__(self, self.mab):
#         i_arm = []
#         coeffs = (1., 1.)
#         mi = None
#         if (len(self.mab.mean_rewards) > 2*len(self.mab.space)):
#             mi = min(self.mab.mean_rewards, key=self.mab.mean_rewards.get)
#             coeff = np.random.rand()*1/np.sqrt(len(self.mab.mean_rewards)/(2*len(self.mab.space)))
#             if (np.random.rand() > coeff):
#                 if (np.random.rand() > coeff):
#                     coeffs = (0., np.random.rand())
#                 else:
#                     coeffs = (coeff, np.random.rand())
#         for i,(p,c,std,lg) in enumerate(self.mab.space):
#             cc = c
#             sstd = std
#             if (mi is not None):
#                 if lg:
#                     cc = np.exp(np.log(c)*coeffs[0] + np.log(mi[i])*(1-coeffs[0]))
#                 else:
#                     cc = c*coeffs[0] + mi[i]*(1-coeffs[0])
#                 sstd = std*coeffs[1]
#             self.all_centers.append(np.log(cc))
#             if lg:
#                 r = np.exp(np.random.randn()*sstd + np.log(cc))
#             else:
#                 r = np.random.randn()*sstd + cc
#             i_arm.append(r)
#         return tuple(i_arm)



class Gaussian_UCB_Continuous_Policy(Continuous_Policy):
    """
    must have array_rewards
    """
    def __init__(self, mab=None, exploit_explore=None, knn_coeff=None, **kargs):
        super(Gaussian_UCB_Continuous_Policy, self).__init__(mab=mab)
        self.exploit_explore = exploit_explore
        self.knn_coeff = knn_coeff
        self.set_params(exploit_explore=exploit_explore, knn_coeff=knn_coeff, **kargs)

    def _set_param(self, k, v):
        if (k == "ee_ratio") or (k == "exploit_explore") or (k=="exploitation_over_exploration_ratio"):
            if v is None:
                v = 0.4
            if (self.exploit_explore != v):
                self.exploit_explore = v
                return True
            return False
        elif (k == "knn_coeff"):
            if v is None:
                v = 1.
            if (self.knn_coeff != v):
                self.knn_coeff = v
                return True
            return False
        else:
            return super(Gaussian_UCB_Continuous_Policy, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "ee_ratio") or (k == "exploit_explore") or (k=="exploitation_over_exploration_ratio"):
            return self.exploit_explore
        elif (k == "knn_coeff"):
            return self.knn_coeff
        else:
            return super(Gaussian_UCB_Continuous_Policy, self)._get_param(k)

    def estimator_train(self, points=None, x_lg=None, points_lg=False):
        # if (len(self.mab.mean_rewards) > 10*len(self.mab.space)):
        sh = self.mab.array_rewards.shape[0]
        mi = self.estimator_test(points=points, x_lg=x_lg, points_lg=points_lg)
        mi = mi/np.std(mi)*0.3
        # weights = self.exploit_explore/np.mean(self.sorted_distance[:,:3], axis=1)
        # dists = self.sorted_distance[:,0]
        # dists /= np.max(dists)/4
        weights = self.exploit_explore*self.sum_weights#/(dists + 0.75)
        #/np.mean(self.sorted_distance[:,:3], axis=1)
        mi -= np.sqrt(2*np.log(sh)/weights)
        # mi = tuple(x_lg_[np.argmin(mi),:])
        return mi
        # else:
        #     return estimator_test(points)

    def estimator_test(self, points=None, x_lg=None, points_lg=False):
        y = self.mab.array_rewards[:,-1]
        lstd = [std for (_,_,std,_) in self.mab.space]
        if x_lg is None:
            x_lg = self.mab.array_rewards[:,:-1].copy()
            for i,(_,_,_,lg) in enumerate(self.mab.space):
                if lg:
                    x_lg[:,i] = np.log(x_lg[:,i])
        x_lg /= lstd
        if (points is None):
            xx_lg = x_lg
        else:
            xx_lg = points.copy()
            if not points_lg:
                for i,(_,_,_,lg) in enumerate(self.mab.space):
                    if lg:
                        xx_lg[:,i] = np.log(xx_lg[:,i])
            xx_lg /= lstd
        sh = x_lg.shape[0]
        # sq = np.maximum(int(self.knn_coeff*np.sqrt(sh)), 10)
        sq = np.maximum(int(self.knn_coeff*(sh**0.75)), 10)
        # return gaussian_average_knn_estimator((x_lg,y), (xx_lg,0), k=sq, slf=self)
        return self._estimator((x_lg,y), (xx_lg,0), k=sq, slf=self)

    def _estimator(self, *args, **kargs):
        return gaussian_average_knn_estimator(*args, **kargs)
                    
        # xx = points
        # x = self.mab.array_rewards[:,:-1]
        # y = self.mab.array_rewards[:,-1]
        # if xx is None:
        #     xx = x
        # sh = x.shape[0]
        # sq = np.maximum(int(self.knn_coeff*np.sqrt(sh)), 10)
        # x_lg = np.zeros(x.shape)
        # xx_lg = np.zeros(xx.shape)
        # for i,(_,_,_,lg) in enumerate(self.mab.space):
        #     if lg:
        #         x_lg[:,i] = np.log(x[:,i])
        #         xx_lg[:,i] = np.log(xx[:,i])
        #     else:
        #         x_lg[:,i] = x[:,i]
        #         xx_lg[:,i] = xx[:,i]
        # lstd = [std for (_,_,std,_) in self.mab.space]
        # # g = Empty_Class()
        # mi = gaussian_average_knn_estimator((x_lg/lstd,y), (xx_lg/lstd,0), k=sq, slf=self)
        # return mi

    def _train(self):
        return self.__call__()

    def __call__(self):
        if (len(self.mab.mean_rewards) > 10*len(self.mab.space)):
            sh = self.mab.array_rewards.shape[0]
            m = np.maximum(sh/8, 10)
            # sq = np.maximum(int(self.knn_coeff*np.sqrt(sh)), 10)
            x_lg = self.mab.array_rewards[:,:-1].copy()
            for i,(_,_,_,lg) in enumerate(self.mab.space):
                if lg:
                    x_lg[:,i] = np.log(x_lg[:,i])
            #     else:
            #         x_lg[:,i] = x[:,i]
            lstd = [std for (_,_,std,_) in self.mab.space]
            rs_std = np.random.randn(m, len(self.mab.space))*lstd
            if (sh < 300):
                xc_ = [[(np.log(i[1]) if i[3] else i[1]) for i in self.mab.space]]
            else:
                xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
            x_lg_ = rs_std + xc_
            mi = self.estimator_train(points=x_lg_, x_lg=x_lg, points_lg=True)
            mi = tuple(x_lg_[np.argmin(mi),:])
            i_arm = tuple([np.exp(m) if lg else m for m,(_,_,_,lg) in zip(mi, self.mab.space)])
        else:
            i_arm = []
            for i,(p,c,std,lg) in enumerate(self.mab.space):
                if lg:
                    i_arm.append(np.exp(np.random.randn()*std + np.log(c)))
                else:
                    i_arm.append(np.random.randn()*std + c)
        return tuple(i_arm)

    # def estimator_best(self, self.mab, x=None):
    #     xx = x
    #     x = np.asarray(self.mab.array_rewards)[:,:-1]
    #     y = np.asarray(self.mab.array_rewards)[:,-1]
    #     if xx is None:
    #         xx = x
    #     sh = x.shape[0]
    #     sq = np.maximum(int(self.knn_coeff*np.sqrt(sh)), 10)
    #     x_lg = np.zeros(x.shape)
    #     xx_lg = np.zeros(x.shape)
    #     for i,(_,_,_,lg) in enumerate(self.mab.space):
    #         if lg:
    #             x_lg[:,i] = np.log(x[:,i])
    #             xx_lg[:,i] = np.log(xx[:,i])
    #         else:
    #             x_lg[:,i] = x[:,i]
    #             xx_lg[:,i] = xx[:,i]
    #     lstd = [std for (_,_,std,_) in self.mab.space]
    #     g = Empty_Class()
    #     mi = gaussian_average_knn_estimator((x_lg/lstd,y), (xx_lg/lstd,0), k=sq, slf=g)
    #     mi = tuple(xx_lg[np.argmin(mi),:])
    #     mi = [(np.exp(mii) if lg else mii) for (mii,(_,_,_,lg)) in zip(mi, self.mab.space)]
    #     return mi
        
    # def __call__(self, self.mab):
    #     i_arm = []
    #     if (self.exploitation_over_exploration_ratio is None):
    #         self.exploitation_over_exploration_ratio = 0.4
    #     if (self.knn_coeff is None):
    #         self.knn_coeff = 1.
    #     # gstd = 0.3
    #     mi = None
    #     if (len(self.mab.mean_rewards) > 10*len(self.mab.space)):
    #         x = np.asarray(self.mab.array_rewards)[:,:-1]
    #         y = np.asarray(self.mab.array_rewards)[:,-1]
    #         sh = x.shape[0]
    #         m = np.maximum(sh/4, 10)
    #         sq = np.maximum(int(self.knn_coeff*np.sqrt(sh)), 10)
    #         # gstd = np.maximum(2.,1.8+x.shape[0]/1000.)/x.shape[0]**0.25
    #         x_lg = np.zeros(x.shape)
    #         for i,(_,_,_,lg) in enumerate(self.mab.space):
    #             if lg:
    #                 x_lg[:,i] = np.log(x[:,i])
    #             else:
    #                 x_lg[:,i] = x[:,i]
    #         lstd = [std for (_,_,std,_) in self.mab.space]
    #         rs_std = np.random.randn(m, len(self.mab.space))*lstd
    #         if (x.shape[0] < 300):
    #             xc_ = [[(np.log(i[1]) if i[3] else i[1]) for i in self.mab.space]]
    #         else:
    #             xc_ = np.mean(x_lg, axis=0)[np.newaxis,:]
    #         x_lg_ = rs_std + xc_
    #         # g = Gaussian_Average_KNN_Estimator(Train_Test([(x_lg/lstd,y),(x_lg_/lstd,0)]), k=sq)
    #         # mi = g.test()
    #         g = Empty_Class()
    #         mi = gaussian_average_knn_estimator((x_lg/lstd,y), (x_lg_/lstd,0), k=sq, slf=g)
    #         mi = mi/np.std(mi)*0.3
    #         coef = self.exploitation_over_exploration_ratio
    #         weights = coef/np.mean(g.sorted_distance[:,:3]**2, axis=1)
    #         mi -= np.sqrt(2*np.log(sh)/weights)
    #         mi = tuple(x_lg_[np.argmin(mi),:])
    #     for i,(p,c,std,lg) in enumerate(self.mab.space):
    #         cc = c
    #         sstd = std
    #         if (mi is not None):
    #             if lg:
    #                 cc = np.exp(mi[i])
    #             else:
    #                 cc = mi[i]
    #             sstd = std*0.
    #         # self.all_centers.append(np.log(cc))
    #         if lg:
    #             r = np.exp(np.random.randn()*sstd + np.log(cc))
    #         else:
    #             r = np.random.randn()*sstd + cc
    #         i_arm.append(r)
    #     return tuple(i_arm)




# arr = np.zeros(10,3)
# np.append(arr, np.zeros(3))


class Gaussian_UCB_2_Continuous_Policy(Gaussian_UCB_Continuous_Policy):

    def _estimator(self, *args, **kargs):
        return gaussian_average_max_knn_estimator(*args, **kargs)
    


class Gaussian_UCB_3_Continuous_Policy(Gaussian_UCB_Continuous_Policy):

    def _estimator(self, *args, **kargs):
        return gaussian_average_median_knn_estimator(*args, **kargs)
    
