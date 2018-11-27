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
from multi_armed_bandit.multi_armed_bandit import *
# from multi_armed_bandit.continuous_multi_armed_bandit import *
from multi_armed_bandit.policy import *
# from multi_armed_bandit.continuous_policy import *

from processing.scale import *



data = Sklearn_Dataset("boston")
n_data = Split_Dataset_N_Parts(data, n_parts=10, seed=0)
tt_data = Join_Dataset_N_Parts(n_data, index=0)
rescale = Scaler_xy(tt_data)

# Compute several regressions, the complete list is in regressor.includes.all_regressors
# regs = [Regressor(rescale, Ridge, regressor_kargs={"alpha":1.}),
#         Regressor(rescale, LinearRegression),
#         Regressor(rescale, BayesianRidge),
#         Regressor(rescale, RandomForestRegressor),
#         Regressor(rescale, SVR(C=1.)),
#         Kernel_Regressor(rescale, SVR, kernel=RBF()),
#         Kernel_Regressor(rescale, KernelRidge, kernel=RBF(gamma=None))]
regs = []
for reg in all_regressors:
    if reg().get_params().has_key("kernel"):
        for ker in all_kernels:
            if isinstance(ker, str):
                regs.append(Kernel_Regressor(rescale, regressor=reg, kernel=ker))
            else:
                regs.append(Kernel_Regressor(rescale, regressor=reg, kernel=ker()))
    else:
        regs.append(Regressor(rescale, regressor=reg))


select = Select_Block(regs, index=0)
run_time = Measure_Time(select)

target = Select_Target(rescale)
error = Prediction_Error_L2(run_time, target)

exc = Ignore_Exception(error, exception=[ValueError, TypeError], default_value=[100,100])
timeout = Timeout(exc, seconds=1, default_value=[100,100])

class Custom(Base_Input_Block):
    def __init__(self, input_block=NoBlock, run_time=NoBlock, **kargs):
        super(Custom, self).__init__(input_block, **kargs)
        self.run_time = run_time
    def compute(self):
        return self._input_block_() + [self.run_time.time]
custom = Custom(timeout, run_time)

def verbose_func():
    args = []
    args += [n_data.get_params("seed")["seed"]]
    args += [tt_data.get_params("index")["index"]]
    args += np.round(custom(), 3).tolist()
    args += [regs[select.get_params("index")["index"]].get_params("name")["name"]]
    print("s {} i {} train : {}   test : {}   time : {}   name : {}".format(*args))
verbose = Verbose(custom, func=verbose_func)

force = Force_Compute(verbose)

class Arm(object):
    def __init__(self, n_data, tt_data, select, index_reg):
        self.n_data = n_data
        self.tt_data = tt_data
        self.select = select
        self.index_reg = index_reg
        self.seed = 0
        self.index_part = -1
    def __call__(self):
        self.index_part += 1
        if (self.index_part >= self.n_data.get_params("n_parts")["n_parts"]):
            self.index_part = 0
            self.seed += 1
        self.n_data.set_params(seed=self.seed)
        self.tt_data.set_params(index=self.index_part)
        self.select.set_params(index=self.index_reg)

arms = [Run_Function_Before(force, func=Arm(n_data, tt_data, select, i)) for i in range(len(regs))]

reward_func = lambda x:x[1] # Since arms returns [train, test, time] and mab reward must be a float
# policy = Uniform_Policy()
policy = Uniform_Threshold_Policy(threshold_max=0.9)
mab = Multi_Armed_Bandit(arms, policy=policy, repeat_max=15, reward_func=reward_func)

mab.train()
mab.test()
mab()






def final_verbose_func():
    print("\n\n")
    order = np.argsort(mab.mean_rewards)
    for i in order:
        rewards = mab.raw_rewards[i]
        #    for i,rewards in enumerate(mab.raw_rewards):
        m_tr, m_te, m_ti, = np.mean(np.asarray(rewards), axis=0).tolist()
        s_tr, s_te, s_ti, = np.std(np.asarray(rewards), axis=0).tolist()
        if (m_te < 10):
            print "train_err : {:<5}+-{:<5},   test_err : {:<5}+-{:<5},   time : {:<5}+-{:<5}   : {}".format(round(m_tr,3), round(s_tr,3), round(m_te,3), round(s_te,3), round(m_ti,3), round(s_ti,3), regs[i].name)

final_verbose = Verbose(mab, func=final_verbose_func)

res = final_verbose()
