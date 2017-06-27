import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import datetime
import random
from scipy.stats.stats import pearsonr

s = pd.Series([1, 2, 3, 4, np.nan, np.nan, np.nan])
print s.isnull().sum()