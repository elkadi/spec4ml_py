#System Modules
import os
import sys
import time
import glob
import random
import warnings
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import ast
#import evaluation_functions
#Visualization Modules
import seaborn as sns
import matplotlib.pyplot as plt
#Statistics Modules
from math import sqrt
from scipy.stats import pearsonr
from copy import copy
#Machine Learning
##TPOT
from tpot import TPOTRegressor
from tpot.export_utils import set_param_recursive
from tpot.builtins import OneHotEncoder, StackingEstimator, ZeroCount
##SkLearn
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.decomposition import FastICA
from sklearn.ensemble import (
    AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import ElasticNetCV, SGDRegressor, RidgeCV
from sklearn.preprocessing import (
    FunctionTransformer, PolynomialFeatures, StandardScaler, Binarizer,
    MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
)

from sklearn.feature_selection import (
    SelectFwe, f_regression, SelectPercentile, VarianceThreshold, SelectFromModel
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import clone
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")



