import numpy as np

from sklearn.metrics.pairwise import euclidean_distances as l2_dist

from utils.base import *



symmetric_distance_l2 = l2_dist
asymmetric_distance_l2 = l2_dist



# TODO do not remove sklearn args like squared
class Symmetric_Distance_L2(Base_Input_Block):
    def compute(self):
        return l2_dist(self._input_block_())



class Asymmetric_Distance_L2(Base_Inputs_Block):
    def compute(self):
        return l2_dist(self._input_block_[0](), self._input_block_[1]())


Symetric_Distance_L2 = Symmetric_Distance_L2
Asymetric_Distance_L2 = Asymmetric_Distance_L2
