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
from plot.grid import *



data = Sklearn_Dataset("boston")
tt_data = Split_Dataset(data, test_size=0.1, seed=None)
rescale = Scaler_xy(tt_data)

reg = Kernel_Regressor(rescale, KernelRidge, kernel=RBF())
reg()

default_alpha = reg.get_params("reg__alpha")
default_gamma = reg.get_params("kernel__gamma")

targets = Select_Target(rescale)
error = Prediction_Error_L2(reg, targets)

force = Force_Compute(error)

def incrementer():
    seed = tt_data.get_params("seed")
    if seed is None:
        tt_data.set_params(seed=None)
    else:
        tt_data.set_params(seed=seed+1)
func_after = Run_Function_After(force, func=incrementer)




# policy = Gaussian_UCB_Continuous_Policy()
# policy = Gaussian_UCB_2_Continuous_Policy()
policy = Gaussian_UCB_3_Continuous_Policy()
policy.set_params(exploit_explore=0.5, knn_coeff=1.)
# policy.exploitation_over_exploration_ratio = 1.
# policy.knn_coeff = 1.
r_f = lambda x:x[1]
space = [((reg, "reg__alpha"), default_alpha, 2., True),
         ((reg, "kernel__gamma"), default_gamma, 2., True)]
# space = [((reg, "reg__alpha"), default_alpha, 2., True)]
rep_min = 1

mab = Continuous_MAB(arm=func_after, policy=policy, space=space, n_max=8000, time_max=3., reward_func=r_f, repeat_min=rep_min)



mab()


import matplotlib.pyplot as plt
plt.scatter(np.log(mab.array_rewards[:,0]), np.log(mab.array_rewards[:,1]), s=50, c=mab.array_rewards[:,-1])
plt.show()


if (len(mab.space) == 1):
    grd = grid(mab.array_rewards[:,:-1], n_parts=1000, log=True, return_log=True)
else:
    grd = grid(mab.array_rewards[:,:-1], n_parts=100, log=True, return_log=True)
    
    
est = mab.policy.estimator_test(points=grd, points_lg=True)

min_point = grd[np.argmin(est),:]
print min_point

if (grd.shape[1] == 1):
    plt.scatter(grd, np.log(est), s=50, c=np.log(est))
    plt.scatter(min_point, np.log(np.min(est)), s=15, c="red")
elif (grd.shape[1] == 2):
    plt.scatter(grd[:,0], grd[:,1], s=50, c=np.log(est))
    plt.scatter(min_point[0], min_point[1], s=15, c="red")
plt.show()
