import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

rng = pd.date_range('1/1/2011', periods=360, freq='1 min')
dat = pd.DataFrame(np.random.rand(len(rng)))
dat['time'] = rng
dat = dat.set_index(pd.to_datetime(dat['time']))
del dat['time']
dat.columns = ['values']

# now check how to integrate the data

df_int = integrate.cumtrapz(dat['values'], dat.index.astype(np.int64) / 10**9, initial=0)

dat = pd.DataFrame(np.empty(len(rng)))
dat['time'] = rng
dat['integral'] = df_int
dat.plot(x='time', y='integral')
plt.show()
#dat['integral'] = df_int