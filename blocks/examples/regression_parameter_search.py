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
from multi_armed_bandit.continuous_multi_armed_bandit import *
# from multi_armed_bandit.policy import *
from multi_armed_bandit.continuous_policy import *

from processing.scale import *


data = Sklearn_Dataset("boston")
tt_data = Split_Dataset(data, test_size=0.1, seed=0)
rescale = Scaler_xy(tt_data)

reg = Kernel_Regressor(rescale, KernelRidge, kernel=RBF())
reg()

default_alpha = reg.get_params("reg__alpha")["reg__alpha"]
default_gamma = reg.get_params("kernel__gamma")["kernel__gamma"]

targets = Select_Target(rescale)
error = Prediction_Error_L2(reg, targets)

force = Force_Compute(error)

def incrementer():
    tt_data.set_params(seed=tt_data.get_params("seed")["seed"]+1)
func_after = Run_Function_After(force, func=incrementer)




policy = Gaussian_UCB_Continuous_Policy()
policy.exploitation_over_exploration_ratio = 0.3
policy.knn_coeff = 1.
r_f = lambda x:x[1]
space = [((reg, "reg__alpha"), default_alpha, 2., True),
         ((reg, "kernel__gamma"), default_gamma, 2., True)]
rep_min = 1
mab = Continuous_MAB(arm=func_after, policy=policy, space=space, n_max=100, reward_func=r_f, repeat_min=rep_min, reward_array=True)



mab()
