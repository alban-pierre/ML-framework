import numpy as np

from utils.base import *
from utils.redirect import *
from dataset.matrix_dataset import *
from dataset.split_dataset import *
from regressor.regressor import *
from regressor.includes import *
from error_measure.prediction_error import *
from processing.scale import *


# Load data -> (x,y)
data = Sklearn_Dataset("boston")
# Split data in 2, with size 90% and 10% -> [(x_train,y_train), (x_test,y_test)]
tt_data = Split_Dataset(data, test_size=0.1, seed=0)
# Rescale data (mean=0, std=1) according to train samples -> [(x_train,y_train), (x_test,y_test)]
rescale = Scaler_xy(tt_data)

# Compute the regression Ridge, according to sklearn Ridge implementation -> [pred_train, pred_test]
reg = Regressor(rescale, Ridge)
# Select the targets -> [y_train, y_test]
targets = Select_Target(rescale)
# Compute the L2 error and average it -> [mean_l2_train_error, mean_l2_test_error]
error = Prediction_Error_L2(reg, targets)

# Compute only the train -> mean_l2_train_error
print error.train()
# Compute train and test, and returns test error -> mean_l2_test_error
print error.test()
# Compute train and test, and returns both errors -> [mean_l2_train_error, mean_l2_test_error]
print error()
