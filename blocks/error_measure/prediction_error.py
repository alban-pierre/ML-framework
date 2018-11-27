from sklearn.metrics import mean_squared_error

from utils.base import *



class Prediction_Error_L2(Base_Inputs_Block):

    def __init__(self, predictor=NoBlock, dataset=NoBlock, average=True, **kargs):
        super(Prediction_Error_L2, self).__init__([predictor, dataset])
        self.params_names = self.params_names.union({"average"})
        self.average = average
        self.set_params(average=average, **kargs)

    def set_params(self, *args, **kargs):
        super(Prediction_Error_L2, self).set_params(*args)
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)

    def _set_param(self, k, v):
        if (k == "average") or (k == "mean"):
            if (self.average != v):
                self.average = v
                return True
            return False
        else:
            if (k == "predictor"):
                k = "input_0"
            elif (k == "dataset"):
                k = "input_1"
            return super(Prediction_Error_L2, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "average") or (k == "mean"):
            return self.average
        else:
            return super(Prediction_Error_L2, self)._get_param(k)

    def compute_error(self, predicted, truth):
         if self.average:
             return mean_squared_error(predicted, truth)
         else:
             return [(y1-y2)**2 for y1, y2 in zip(predicted, truth)]

    def compute(self):
        predicted = self._input_block_[0]()
        truth = self._input_block_[1]()
        if isinstance(predicted, tuple) or isinstance(predicted, list):
            return [self.compute_error(p, t) for p,t in zip(predicted, truth)]
        else:
            return self.compute_error(predicted, truth)
