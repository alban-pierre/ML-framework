import numpy as np 
from torch.utils.data import Dataset



class Split_Pytorch_Dataset(Dataset):

    def __init__(self, dataset, indexes=None, split_func=None, test_size=0.1, seed=None):
        self.dataset = dataset
        self.split_func = split_func
        self.test_size = test_size
        self.seed = seed
        if (indexes is None):
            x = np.arange(len(self.dataset))
            if hasattr(self.split_func, "feature") and (self.split_func.feature == ('y', 0)):
                y = np.asarray([i[1] for i in self.dataset])
            else:
                y = np.arange(len(self.dataset))
            self.indexes, self.indexes_test = self.split_func(x, y, test_size=test_size, seed=seed)
        else:
            self.indexes = indexes
            self.indexes_test = None

    def __len__(self):
        return self.indexes.shape[0]

    def __getitem__(self, idx):
        return self.dataset[self.indexes[idx]]
    
