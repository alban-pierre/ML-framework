import time
import numpy as np


class Uniform_MAB:
    """
    Basic multi armed bandit class, should be the parent of the other MAB classes
    It pulls the arms one after the other, in a uniform way (0,1,2,3,0,1,2,3,0,1,...)
    The list of rewards is in self.list_rewards and the mean of rewards is in self.mean_rewards
    The list of other data stored is in self.other_data
    """
    

    def __init__(self, n_arms=1, n_max=-1, time_max=-1):
        """
        n_arms : (int) = 1 : the number of arms
        n_max : (int) = -1 : maximum number of arms pulled
        time_max : (float) = -1 : maximum time between setting this instance and the last arm pulled
        You can define these attributes later with the function set
        Before pulling arms, n_arms must be defined, and either n_max or time_max must be defined
        """
        self.n_arms = None
        self.n_max = None
        self.time_max = None
        self.set(n_arms, n_max, time_max)


        
    def set(self, n_arms=-1, n_max=-1, time_max=-1):
        """
        Set the inner parameters of this MAB class
        n_arms : (int) = -1 : the number of arms
        n_max : (int) = -1 : maximum number of arms pulled
        time_max : (float) = -1 : maximum time between setting this instance and the last arm pulled
        You can define these attributes later by using this same function
        Before pulling arms, n_arms must be defined, and either n_max or time_max must be defined
        """
        if (n_arms is not None) and (n_arms > 0):
            self.n_arms = n_arms
        if (n_max is None) or (n_max > 0):
            self.n_max = n_max
        if (time_max is None) or (time_max > 0):
            self.time_max = time_max
        if (self.n_arms is None):
            self.n_arms = 1
        self.list_next = range(self.n_arms)
        self.n = 0
        self.time = time.time()
        self.list_rewards = [[] for i in range(self.n_arms)]
        self.mean_rewards = [0 for i in range(self.n_arms)]
        self.other_data = [[] for i in range(self.n_arms)]
        

        
    def next_arm(self):
        """
        Return the arm index of the next arm that should be pulled
        Return None if the maximum amount of arm pulled is reached
        Return None if the maximum amount of time pulling arms is reached
        While the function update_reward is not used (or skip_arm), it returns the same index
        """
        if (self.n_max is not None) and (self.n >= self.n_max):
            return None
        if (self.time_max is not None) and (time.time() - self.time >= self.time_max):
            return None
        return self._next_arm()


    
    def _next_arm(self):
        """
        You should use "next_arm" instead
        Return the arm index of the next arm that should be pulled
        It makes no verification concerning the maximum amount of arms pulled or maximum of time spent
        While the function update_reward is not used (or skip_arm), it returns the same index
        """
        if len(self.list_next) == 0:
            self.list_next = range(self.n_arms)
        return self.list_next[0]



    def skip_arm(self):
        """
        Skip one arm from being pulled
        """
        self._next_arm()
        self.list_next = self.list_next[1:]

    

    def update_reward(self, reward, arm=None, other_data=None):
        """
        Update the reward obtained while pulling one arm
        reward : (float) : the reward obtained
        arm : (int) = None : the arm index pulled, default is self._next_arm()
        other_data : (*) = None : other data that we want to store per arm, ie like the reward
        """
        if arm is None:
            arm = self._next_arm()
        self.list_next = self.list_next[1:]
        self.n += 1
        self.list_rewards[arm].append(reward)
        if (reward is None):
            self.mean_rewards[arm] = None
        elif (self.mean_rewards[arm] is not None):
            self.mean_rewards[arm] = np.mean(self.list_rewards[arm])
        self.other_data[arm].append(other_data)
        
