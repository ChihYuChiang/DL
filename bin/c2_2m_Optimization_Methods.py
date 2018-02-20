import numpy as np
import matplotlib.pyplot as plt
from c2_1_reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from c2_1_reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from c2_1_testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) #set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


'''
--------------------------------------------------
Model
--------------------------------------------------
'''