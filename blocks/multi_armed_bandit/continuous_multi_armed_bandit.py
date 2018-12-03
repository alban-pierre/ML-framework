# import os
import time
import numpy as np

from utils.base import *



class Continuous_MAB(Base_Input_Block):
    """
    Basic multi armed bandit class, should be the parent of the other MAB classes
    It pulls the arms one after the other, in a uniform way (0,1,2,3,0,1,2,3,0,1,...)
    The list of rewards is in self.list_rewards and the mean of rewards is in self.mean_rewards
    The list of other data stored is in self.other_data
    """

    def __init__(self, arm=NoBlock, policy=None, space=None, n_max=None, time_max=None, reward_func=None, start_time=None, repeat_min=None, inversed=False, **kargs):
        """
        n_arms : (int) = 1 : the number of arms
        repeat_max : (int) = -1 : stops when one arm is pulled more times than repeat_max
        n_max : (int) = -1 : stops when the total number of arms pulled is over n_max
        time_max : (float) = -1 : maximum time between setting this instance and the last arm pulled
        You can define these attributes later with the function set_params
        Before pulling arms, n_arms must be defined, and also either repeat_max or n_max or time_max
        """
        super(Continuous_MAB, self).__init__(arm)
        self.params_names = self.params_names.union({"repeat_max", "n_max", "time_max"})
        self.params_names = self.params_names.union({"reward_func", "start_time", "repeat_min"})
        if policy is None:
            policy = Uniform_Policy()
        self.set_params(reward_func=reward_func, repeat_min=repeat_min, inversed=inversed)
        self.set_params(policy=policy, space=space, n_max=n_max, time_max=time_max)
        self.set_params(start_time=start_time, **kargs)
        # self.set_params(reward_array=reward_array, start_time=start_time, **kargs)

    def set_params(self, **kargs):
        changed = False
        for k,v in kargs.iteritems():
            changed = self._set_param(k, v) or changed
        if changed:
            self.set_changed_here(True)
            self.arm = self.input_block
            self.reset()

    def _set_param(self, k, v):
        if (k == "policy"):
            self.policy = v
            self.policy.set_params(mab=self)
            return False
        elif (k == "space"):
            self.space = v
            return True
        elif (k == "n_max"):
            if (v > 0) or (v is None):
                self.n_max = v
                self.set_changed_here_train(True)
                return False
            else:
                print("MAB Warning : incorrect value for n_max")
                return False
        elif (k == "time_max"):
            if (v > 0) or (v is None):
                self.time_max = v
                self.set_changed_here_train(True)
                return False
            else:
                print("MAB Warning : incorrect value for time_max")
                return False
        elif (k == "time") or (k == "start_time"):
            if (v is None):
                self.time = time.time()
            else:
                self.time = v
                self.set_changed_here_train(True)
                return False
        elif (k == "reward_func") or (k == "reward_function"):
            self.reward_func = v
            return True
        elif (k == "repeat_min"):
            if (v > 0) or (v is None):
                self.repeat_min = v
                self.set_changed_here_train(True)
                return False
            else:
                print("MAB Warning : incorrect value for repeat_min")
                return False
        elif (k == "inverse") or (k == "inversed"):
            self.inversed = v
            return False
        # elif (k == "reward_array"):
        #     self.reward_array = v
        #     return False
        else:
            if (k[:4] == "arms"):
                k = "input" + k[4:]
            elif (k[:3] == "arm"):
                k = "input" + k[3:]
            res = self.policy._set_param(k, v)
            if not res:
                return super(Continuous_MAB, self)._set_param(k, v)

    def _get_param(self, k):
        if (k == "policy"):
            return self.policy
        if (k == "space"):
            return self.space
        elif (k == "n_max"):
            return self.n_max
        elif (k == "time_max"):
            return self.time_max
        elif (k == "time") or (k == "start_time"):
            return self.time
        elif (k == "reward_func") or (k == "reward_function"):
            return self.reward_func
        elif (k == "repeat_min"):
            return self.repeat_min
        # elif (k == "reward_array"):
        #     return self.reward_array
        else:
            res = self.policy._get_param(k)
            if res is NoParam:
                return super(Continuous_MAB, self)._get_param(k)
            else:
                return res

    def changed_train(self):
        return self._changed_here_train

    def changed_test(self):
        return self._changed_here_test or self.changed_train()

    def changed_call(self):
        return self.changed_test()

    def changed(self):
        return self.changed_test()

    def reset(self):
        self.set_changed_here(True)
        self.list_next = []
        self.n = 0
        self.list_rewards = {}
        self.mean_rewards = {}
        self.array_rewards = np.asarray([])
        self._array_rewards = np.asarray([])
        if self.reward_func is None:
            self.raw_rewards = self.list_rewards
        else:
            self.raw_rewards = {}
        self.output_train = self.mean_rewards
        self.min_reward = None
        self.max_reward = None

    def next_arm(self):
        """
        Return the arm index of the next arm that should be pulled
        Return None if the maximum amount of arm pulled is reached
        Return None if the maximum amount of time pulling arms is reached
        While the function update_reward is not used (or skip_arm), it returns the same index
        """
        if self._changed_here_train:
            if (self.n_max is not None) and (self.n >= self.n_max):
                self._update_changed_train()
                return None
            if (self.time_max is not None) and (time.time() - self.time >= self.time_max):
                self._update_changed_train()
                return None
            na = self._next_arm()
            self.set_changed_here_test(True)
            return na
        return None

    def _next_arm(self):
        """
        You should use "next_arm" instead
        Return the arm index of the next arm that should be pulled
        It makes no verification concerning the maximum amount of arms pulled or maximum of time spent
        While the function update_reward is not used (or skip_arm), it returns the same index
        """
        if (len(self.list_next) == 0):
            res = self.policy()
            if isinstance(res, list):
                if self.repeat_min is None:
                    self.list_next = res
                else:
                    self.list_next = list(sum(zip(*[range(res) for i in range(self.repeat_min)]), ()))
            else:
                if self.repeat_min is None:
                    self.list_next.append(res)
                else:
                    self.list_next += [res]*self.repeat_min
        return self.list_next[0]

    def skip_arm(self):
        """
        Skip one arm from being pulled
        """
        self._next_arm()
        self.list_next = self.list_next[1:]

    def update_reward(self, reward, i_arm=None):
        """
        Update the reward obtained while pulling one arm
        reward : (float) : the reward obtained
        arm : (int) = None : the arm index pulled, default is self._next_arm()
        other_data : (*) = None : other data that we want to store per arm, ie like the reward
        """
        if i_arm is None:
            i_arm = self._next_arm()
        self.skip_arm()
        self.n += 1
        if (self.reward_func is None):
            if not self.list_rewards.has_key(i_arm):
                self.list_rewards[i_arm] = []
            self.list_rewards[i_arm].append(reward)
        else:
            if not self.list_rewards.has_key(i_arm):
                self.raw_rewards[i_arm] = []
                self.list_rewards[i_arm] = []
            self.raw_rewards[i_arm].append(reward)
            self.list_rewards[i_arm].append(self.reward_func(reward))
        self.mean_rewards[i_arm] = np.mean(self.list_rewards[i_arm])
        # if self.reward_array:
        #     self.array_rewards.append(list(i_arm) + [self.list_rewards[i_arm][-1]])
        sh = self.array_rewards.shape
        if (self.array_rewards.shape[0] == self._array_rewards.shape[0]):
            if (sh[0] == 0):
                self._array_rewards = np.asarray(list(i_arm) + [self.list_rewards[i_arm][-1]])
                self._array_rewards = self._array_rewards[np.newaxis,:]
                self.array_rewards = self._array_rewards
            else:
                self._array_rewards = np.concatenate([self._array_rewards, np.zeros(sh)])
                self._array_rewards[sh[0],:] = list(i_arm) + [self.list_rewards[i_arm][-1]]
                self.array_rewards = self._array_rewards[:sh[0]+1,:]
        else:
            self._array_rewards[sh[0],:] = list(i_arm) + [self.list_rewards[i_arm][-1]]
            self.array_rewards = self._array_rewards[:sh[0]+1,:]
        if (self.min_reward is None) or (self.min_reward > self.list_rewards[i_arm][-1]):
            self.min_reward = self.list_rewards[i_arm][-1]
        if (self.max_reward is None) or (self.max_reward < self.list_rewards[i_arm][-1]):
            self.max_reward = self.list_rewards[i_arm][-1]
        self.policy.update_reward(i_arm, self.list_rewards[i_arm][-1])

    def pull(self):
        """
        Pull one arm
        """
        i_arm = self.next_arm()
        if i_arm is None:
            return False
        for (p,c,std,lg),v in zip(self.space, i_arm):
            if isinstance(p, tuple) and isinstance(p[0], Base_Block) and isinstance(p[1], str):
                p[0].set_params(**{p[1]:v})
            else:
                p(v)
        reward = self.arm()
        self.update_reward(reward, i_arm)
        return True

    def _train(self):
        while (self.pull()):
            pass
        return self.mean_rewards

    def _test(self):
        self.train()
        return np.min(self.mean_rewards)

    def _call(self):
        self.test()
        return [self.output_train, self.output_test]



# Continuous_MAB = Continuous_Multi_Armed_Bandit
Continuous_Multi_Armed_Bandit = Continuous_MAB
