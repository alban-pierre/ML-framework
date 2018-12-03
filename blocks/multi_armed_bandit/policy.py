from utils.base import *

# TODO https://github.com/bgalbraith/bandits/blob/master/bandits/policy.py



class Policy(Base_Start_Block):

    def __init__(self, mab=None, inversed=False, **kargs):
        super(Policy, self).__init__()
        self.params_names = self.params_names.union({"mab", "inversed"})
        self.mab = mab
        self.inversed = inversed
        self.set_params(mab=mab, inversed=inversed, **kargs)
        
    # def set_params(self, **kargs):
    #     for k,v in kargs.iteritems():
    #         self._set_param(k, v)

    def _set_param(self, k, v):
        if (k == "mab"):
            if (self.mab is not v):
                self.mab = v
                return True
            return False
        elif (k == "inverse") or (k == "inversed"):
            if (self.inversed != v):
                self.inversed = v
                return True
            return False
        else:
            return super(Policy, self)._set_param(k, v)
            
    def _get_param(self, k):
        if (k == "mab"):
            return self.mab
        elif (k == "inverse") or (k == "inversed"):
            return self.inversed
        else:
            return super(Policy, self)._get_param(k)

    # def get_params(self, *args):
    #     if args:
    #         return {k:self._get_param(k) for k in args if k in self.params_names}
    #     else:
    #         return {k:self._get_param(k) for k in self.params_names}

    # def _get_param(self, k):
    #     return None

    def changed_train(self):
        if self.changed_here_train:
            return True
        elif self.mab is None:
            return True
        return self.mab.changed_train()
    
    def changed_test(self):
        if self.changed_here_test:
            return True
        elif self.mab is None:
            return True
        return self.mab.changed_test()
    
    def changed_call(self):
        return True
    
    def update_reward(self, i_arm, reward):
        pass

    def estimator_train(self, i_arms=None):
        return self.estimator_test(i_arms=i_arms)
        
    def estimator_test(self, i_arms=None):
        if isinstance(i_arms, None):
            return self.mab.mean_rewards
        elif isinstance(i_arms, list):
            return [self.mab.mean_rewards[i] for i in i_arms]
        else:
            return self.mab.mean_rewards[i_arms]

    def _train(self):
        return self.__call__()
        # if inversed:
        #     return np.argmax(self.estimator_train())
        # else:
        #     return np.argmin(self.estimator_train())
    
    def _test(self):
        if inversed:
            return np.argmax(self.estimator_test())
        else:
            return np.argmin(self.estimator_test())
    
    def __call__(self):
        # return self._train()
        if inversed:
            return np.argmax(self.estimator_train())
        else:
            return np.argmin(self.estimator_train())



class Uniformize_Policy(Policy):
    """
    Return the arm with the minimum number of pulls
    """
    def __call__(self):
        return np.argmin([len(m) for m in self.mab.list_rewards])



class Uniform_Policy(Policy):
    """
    Return the list of all arms, therefore preserving difference of pulls
    """
    def _train(self):
        return np.argmin([len(m) for m in self.mab.list_rewards])
    
    def __call__(self):
        return range(self.mab.n_arms)



class Uniform_Threshold_Policy(Policy):
    
    def __init__(self, mab=None, threshold_min=None, threshold_max=None, **kargs):
        super(Uniform_Threshold_Policy, self).__init__()
        self.params_names = self.params_names.union({"threshold_min", "threshold_max"})
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.set_params(mab=mab, threshold_min=threshold_min, threshold_max=threshold_max, **kargs)

    def _set_param(self, k, v):
        if (k == "threshold_min") or (k == "th_min") or (k == "min"):
            if (v != self.threshold_min):
                self.threshold_min = v
                return True
            return False
        elif (k == "threshold_max") or (k == "th_max") or (k == "max"):
            if (v != self.threshold_max):
                self.threshold_max = v
                return True
            return False
        else:
            return super(Uniform_Threshold_Policy, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "threshold_min") or (k == "th_min") or (k == "min"):
            return self.threshold_min
        elif (k == "threshold_max") or (k == "th_max") or (k == "max"):
            return self.threshold_max
        else:
            return super(Uniform_Threshold_Policy, self)._get_param(k)

    def _train(self):
        li_arms = self.__call__()
        return li_arms[np.argmin([len(self.mab.list_rewards[i]) for i in li_arms])]
        
    def __call__(self):
        if (self.threshold_min is None):
            if (self.threshold_max is None):
                return range(self.mab.mean_rewards)
            else:
                return [i for i,r in enumerate(self.mab.mean_rewards) if (len(self.mab.list_rewards) == 0) or (r <= self.threshold_max)]
        else:
            if (self.threshold_max is None):
                return [i for i,r in enumerate(self.mab.mean_rewards) if (len(self.mab.list_rewards) == 0) or (r >= self.threshold_min)]
            else:
                return [i for i,r in enumerate(self.mab.mean_rewards) if (len(self.mab.list_rewards) == 0) or ((r >= self.threshold_min) and (r <= self.threshold_max))]


    # def __call__(self):
    #     if (self.threshold_min is None):
    #         if (self.threshold_max is None):
    #             return range(mab.mean_rewards)
    #         else:
    #             return [i for i,r in enumerate(mab.mean_rewards) if (len(mab.list_rewards) == 0) or (r <= self.threshold_max)]
    #     else:
    #         if (self.threshold_max is None):
    #             return [i for i,r in enumerate(mab.mean_rewards) if (len(mab.list_rewards) == 0) or (r >= self.threshold_min)]
    #         else:
    #             return [i for i,r in enumerate(mab.mean_rewards) if (len(mab.list_rewards) == 0) or ((r >= self.threshold_min) and (r <= self.threshold_max))]



class Uniformize_Threshold_Policy(Uniform_Threshold_Policy):

    def _train(self):
        return self.__call__()

    def __call__(self):
        li_arms = super(Uniformize_Threshold_Policy, self).__call__()
        # if (self.threshold_min is None):
        #     if (self.threshold_max is None):
        #         res = range(self.mab.mean_rewards)
        #     else:
        #         res = [i for i,r in enumerate(self.mab.mean_rewards) if (len(self.mab.list_rewards) == 0) or (r <= self.threshold_max)]
        # else:
        #     if (self.threshold_max is None):
        #         res = [i for i,r in enumerate(self.mab.mean_rewards) if (len(self.mab.list_rewards) == 0) or (r >= self.threshold_min)]
        #     else:
        #         res = [i for i,r in enumerate(self.mab.mean_rewards) if (len(self.mab.list_rewards) == 0) or ((r >= self.threshold_min) and (r <= self.threshold_max))]
        # return res[np.argmin([len(self.mab.list_rewards[i]) for i in res])]
        return li_arms[np.argmin([len(self.mab.list_rewards[i]) for i in li_arms])]
