import numpy as np

from utils.base import *



def grid(points, n_parts=100, log=False, return_log=False, slf=Empty_Class()):
    x = points
    sh = x.shape[1]
    # n_parts = self.n_parts
    if not isinstance(n_parts, list):
        n_parts = [n_parts]*sh
    # log = self.log
    if not isinstance(log, list):
        log = [log]*sh
    slf.mi = np.min(x, axis=0)
    slf.ma = np.max(x, axis=0)
    if any(log):
        slf.mi = np.asarray([np.log(m) if lg else m for m,lg in zip(slf.mi, log)])
        slf.ma = np.asarray([np.log(m) if lg else m for m,lg in zip(slf.ma, log)])
    slf.steps = (slf.ma - slf.mi)/(n_parts)
    slf.ma += slf.steps/2
    slf.grd = np.mgrid[tuple(slice(*v) for v in zip(slf.mi, slf.ma, slf.steps))]
    if not return_log and any(log):
        for i,lg in enumerate(log):
            if lg:
                slf.grd[i] = np.exp(slf.grd[i])
    slf.points = slf.grd.reshape(sh,-1).T
    return slf.points

    

class Grid(Base_Input_Block):

    def __init__(self, input_block=NoBlock, n_parts=100, log=False, return_log=False, **kargs):
        super(Grid, self).__init__(input_block, **kargs)
        self.n_parts = n_parts
        self.log = log
        self.return_log = return_log
        
    def compute(self):
        return grid(self._input_block_(), self.n_parts, self.log, self.return_log, self)
        # x = self._input_block_()
        # n_parts = self.n_parts
        # if not isinstance(n_parts, list):
        #     n_parts = [n_parts]*x.shape[1]
        # log = self.log
        # if not isinstance(log, list):
        #     log = [log]*x.shape[1]
        # self.mi = np.min(x, axis=0)
        # self.ma = np.max(x, axis=0)
        # if any(log):
        #     mi = [np.log(m) if lg else m for m,lg in zip(self.mi, log)]
        #     ma = [np.log(m) if lg else m for m,lg in zip(self.ma, log)]
        # self.steps = (ma - mi) / self.n_parts
        # ma += steps/2
        # grid = np.mgrid[tuple(slice(*v) for v in zip(mi, ma, self.steps))]

            

