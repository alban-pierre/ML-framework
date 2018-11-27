from utils.base import *



class Policy(object):

    def __init__(self):
        self.params_names = {}
    
    def set_params(self, **kargs):
        for k,v in kargs.iteritems():
            self._set_param(k, v)

    def _set_param(self, k, v):
        return False
            
    def get_params(self, *args):
        if args:
            return {k:self._get_param(k) for k in args if k in self.params_names}
        else:
            return {k:self._get_param(k) for k in self.params_names}

    def _get_param(self, k):
        return None



class Uniform_Policy(Policy):
    
    def __call__(self, mab):
        return range(mab.n_arms)


    
class Uniform_Threshold_Policy(Policy):
    
    def __init__(self, threshold_min=None, threshold_max=None):
        self.params_names = set({"threshold_min", "threshold_max"})
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

    def _set_param(self, k, v):
        if (k == "threshold_min") or (k == "th_min") or (k == "min"):
            if (v != self.threshold_min):
                self.threshold_min = v
                return True
        elif (k == "threshold_max") or (k == "th_max") or (k == "max"):
            if (v != self.threshold_max):
                self.threshold_max = v
                return True

    def _get_param(self, k):
        if (k == "threshold_min") or (k == "th_min") or (k == "min"):
            return self.threshold_min
        elif (k == "threshold_max") or (k == "th_max") or (k == "max"):
            return self.threshold_max
        return NoParam
    
    def __call__(self, mab):
        if (self.threshold_min is None):
            if (self.threshold_max is None):
                return range(mab.mean_rewards)
            else:
                return [i for i,r in enumerate(mab.mean_rewards) if (len(mab.list_rewards) == 0) or (r <= self.threshold_max)]
        else:
            if (self.threshold_max is None):
                return [i for i,r in enumerate(mab.mean_rewards) if (len(mab.list_rewards) == 0) or (r >= self.threshold_min)]
            else:
                return [i for i,r in enumerate(mab.mean_rewards) if (len(mab.list_rewards) == 0) or ((r >= self.threshold_min) and (r <= self.threshold_max))]

