import pandas as pd
import numpy as np
import timeit
np.random.seed(0)
import datetime
import os
#rng = pd.date_range(start='2011/09/09 10:00:00', end='2011/09/09 12:00:00', freq='50 ms')
#print rng
st = "2016-04-01 00:00:00.000"

dt = datetime.datetime.strptime(st, '%Y-%m-%d %H:%M:%S.%f')
print dt.strftime('%Y-%m-%d')