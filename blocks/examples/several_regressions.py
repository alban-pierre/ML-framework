import numpy as np

from utils.base import *
# # from utils.custom import *
from utils.redirect import *
# from utils.pipeline import *
from dataset.matrix_dataset import *
from dataset.split_dataset import *
from regressor.regressor import *
from regressor.includes import *
from error_measure.prediction_error import *
# from utils.decorators import *
# from multi_armed_bandit.multi_armed_bandit import *
# from multi_armed_bandit.continuous_multi_armed_bandit import *
# from multi_armed_bandit.policy import *
# from multi_armed_bandit.continuous_policy import *

from processing.scale import *


data = Sklearn_Dataset("boston")
tt_data = Split_Dataset(data, test_size=0.1, seed=0)
rescale = Scaler_xy(tt_data)

# Compute several regressions, the complete list is in regressor.includes.all_regressors
regs = [Regressor(rescale, Ridge, regressor_kargs={"alpha":1.}),
        Regressor(rescale, LinearRegression),
        Regressor(rescale, BayesianRidge),
        Regressor(rescale, RandomForestRegressor),
        Regressor(rescale, SVR(C=1.)),
        Kernel_Regressor(rescale, SVR, kernel=RBF()),
        Kernel_Regressor(rescale, KernelRidge, kernel=RBF(gamma=None))]
targets = Select_Target(rescale)
errors = [Prediction_Error_L2(reg, targets) for reg in regs]

# Here we only stack the outputs
stack = Stack_Output(errors)
stack()





###########################
# There are several choices possible, here is another more complicated
###########################

import numpy as np

from utils.base import *
from utils.custom import *
from utils.redirect import *
# from utils.pipeline import *
from dataset.matrix_dataset import *
from dataset.split_dataset import *
from regressor.regressor import *
from regressor.includes import *
from error_measure.prediction_error import *
from utils.decorators import *
# from multi_armed_bandit.multi_armed_bandit import *
# from multi_armed_bandit.continuous_multi_armed_bandit import *
# from multi_armed_bandit.policy import *
# from multi_armed_bandit.continuous_policy import *

from processing.scale import *



data = Sklearn_Dataset("boston")
tt_data = Split_Dataset(data, test_size=0.1, seed=0)
rescale = Scaler_xy(tt_data)

# Compute several regressions, the complete list is in regressor.includes.all_regressors
regs = [Regressor(rescale, Ridge, regressor_kargs={"alpha":1.}),
        Regressor(rescale, LinearRegression),
        Regressor(rescale, BayesianRidge),
        Regressor(rescale, RandomForestRegressor),
        Regressor(rescale, SVR(C=1.)),
        Kernel_Regressor(rescale, SVR, kernel=RBF()),
        Kernel_Regressor(rescale, KernelRidge, kernel=RBF(gamma=None))]

select = Select_Block(regs, index=0)
run_time = Measure_Time(select)

target = Select_Target(rescale)
error = Prediction_Error_L2(run_time, target)

custom = Custom_Input_Block((lambda err,t : err + [t.time]), error, {"t":run_time})
# or
# class Custom(Base_Input_Block):
#     def compute(self):
#         return self._input_block_() + [run_time.time]
# custom = Custom(error)

def verbose_func():
    print("train : {}   test : {}   time : {}".format(*np.round(custom(), 3).tolist()))
verbose = Verbose(custom, func=verbose_func)

force = Force_Compute(verbose)

def incrementer():
    select.set_params(index=select.get_params("index")+1)
    # select.set_params(index=select.get_index()+1)
    # select.set_params(index=select.get_params("index")["index"]+1)
func_after = Run_Function_After(force, func=incrementer)

rescale()
[func_after() for _ in regs]
